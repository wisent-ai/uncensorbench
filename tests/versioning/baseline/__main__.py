"""Generate ``released-surface.json`` from the best reachable artifact.

The baseline describes the version **actually published**, never the version
the working tree declares: every version decision is measured against it.
Run from the repository root as ``python3 tests/versioning/baseline``; the
reader it uses is the product's own ``uncensorbench.surface``. Until
2026-09-06 this was ``scripts/baseline.py``; the workflow that gates versions
ran a copy written by heredoc for four days after that directory was removed,
and the README kept naming the deleted file.

Tier preference, best first (ADOPTING.md): the PyPI sdist, the PyPI wheel, a
GitHub release, the best git tag as an archive, and the working tree when
nothing was ever published. Every absence is read from the answer's content,
in three states -- published, absent, unproven -- so a network that did not
answer is never mistaken for an index that has nothing.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import tempfile
from pathlib import Path

# The product package is read from the checkout this generator sits in, so the
# reader and the baseline it produces are the same revision.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from constants import KNOWN_TIERS, PUBLISHED, PYPI_TIERS, UNPROVEN  # noqa: E402
from pypi import (  # noqa: E402
    assert_control, declared_version, fail, read_pypi, recover_from_pypi, resolve_project,
)
from tiers import best_tag, gh_releases_present, head_baseline  # noqa: E402
from uncensorbench import surface  # noqa: E402


def generate(root):
    project = resolve_project(root)
    state, document = read_pypi(project)
    assert_control()
    if state == UNPROVEN:
        fail(
            "PyPI neither named '"
            + project
            + "' nor stated it is absent, while the control lookup succeeded; the"
            " answer about this project is unproven, so no baseline may be written"
        )
    if state == PUBLISHED:
        with tempfile.TemporaryDirectory() as scratch:
            return recover_from_pypi(document, Path(scratch))
    if gh_releases_present(root):
        fail(
            "PyPI serves nothing but this repository has GitHub Releases, which"
            " outrank both git-archive and head; recover the baseline from the"
            " release asset rather than letting this generator degrade past it"
        )
    from_tag = best_tag(root)
    if from_tag is not None:
        return from_tag
    return head_baseline(root)


def claims_pypi(marker):
    tier, _separator, _rest = marker.partition(":")
    if tier not in KNOWN_TIERS:
        fail(
            "unknown baseline marker tier '"
            + tier
            + "'; known tiers are "
            + ", ".join(KNOWN_TIERS)
        )
    return tier in PYPI_TIERS


def mirror_version(root, package):
    """``__version__`` as the package's ``__init__`` declares it, read with ast."""
    init = root / package / "__init__.py"
    if not init.is_file():
        return None
    tree = surface.parse_module(init, False)
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
    return None


def unambiguous_declared_version(root):
    """The version the product declares, refusing if it declares two.

    ``[project] version`` is canonical -- setuptools stamps it into PKG-INFO and
    into the sdist and wheel filenames -- but ``uncensorbench.__version__`` is a
    second declaration a consumer can read at runtime. While the two disagree,
    ``--current`` has no single answer, so the gate must say so rather than pick.
    """
    declared = declared_version(root)
    if not declared:
        fail("pyproject.toml declares no [project] version")
    package = surface.resolve_package(root, None)
    mirror = mirror_version(root, package)
    if mirror is not None and mirror != declared:
        fail(
            "pyproject.toml declares "
            + declared
            + " but "
            + package
            + "/__init__.py declares __version__ = "
            + mirror
            + "; the product states two versions, so the version it declares is"
            " ambiguous. Make them agree; do not let a check choose."
        )
    return declared


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate released-surface.json from the best reachable artifact."
    )
    parser.add_argument("--root", type=Path, default=here.parents[2])
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the candidate baseline instead of writing the committed file",
    )
    parser.add_argument(
        "--claims-pypi",
        metavar="MARKER",
        help="print yes or no: does this marker assert a PyPI release?",
    )
    parser.add_argument(
        "--project-name",
        action="store_true",
        help="print the distribution name the manifest declares",
    )
    parser.add_argument(
        "--declared-version",
        action="store_true",
        help="print the version the product declares, refusing if it declares two",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if args.claims_pypi:
        print("yes" if claims_pypi(args.claims_pypi) else "no")
        return
    if args.project_name:
        print(resolve_project(root))
        return
    if args.declared_version:
        print(unambiguous_declared_version(root))
        return

    baseline = generate(root)
    rendered = json.dumps(baseline, indent="  ") + "\n"
    if args.stdout:
        sys.stdout.write(rendered)
        return
    target = root / "released-surface.json"
    target.write_text(rendered, encoding="utf-8")
    print("baseline: wrote " + str(target), file=sys.stderr)


if __name__ == "__main__":
    main()
