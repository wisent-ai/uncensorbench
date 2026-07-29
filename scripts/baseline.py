#!/usr/bin/env python3
"""Generate ``released-surface.json`` from the best reachable artifact.

The baseline describes the version **actually published**, never the version
``pyproject.toml`` declares. The moment somebody bumps ahead of a release,
looking up the declared version returns nothing and a naive generator degrades
to ``head:<sha>``, throwing away the real published baseline -- after which
every later comparison is measured against the wrong artifact, quietly. So the
registry is asked what the newest published version is, and *that* version's
artifact is recovered.

Tier preference, best first (ADOPTING.md):

    pypi-sdist:<filename>    a published sdist
    pypi-wheel:<filename>    a published wheel
    gh-release:<tag>         an asset on a GitHub Release
    git-archive:<tag>        a tag, reproduced with git archive
    head:<full sha>          the working revision -- last resort

``--claims-pypi <marker>`` answers whether a marker asserts a PyPI release, so
the workflow's bidirectional honesty guard and this generator agree through a
named constant instead of through prose that drifts.

Every absence here is read from the answer's *content*, in three states --
named, stated-absent, unproven -- because a client that could not reach the
index fails exactly like one that reached it and found nothing, and the wrong
reading is the passing one. Each lookup is paired with a positive control run
through the same URL spelling against a project the index certainly serves, so
a broken expression blames the check rather than the registry.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import surface

PYPI_JSON = "https://pypi.org/pypi/{project}/json"
CONTROL_PROJECT = "pip"

SDIST_TIER = "pypi-sdist"
WHEEL_TIER = "pypi-wheel"
GH_RELEASE_TIER = "gh-release"
GIT_ARCHIVE_TIER = "git-archive"
HEAD_TIER = "head"

#: Markers asserting "this exact version is served by PyPI". Every other tier
#: asserts the opposite -- that PyPI serves this project not at all -- because
#: any PyPI release would outrank it.
PYPI_TIERS = (SDIST_TIER, WHEEL_TIER)

KNOWN_TIERS = (
    SDIST_TIER,
    WHEEL_TIER,
    GH_RELEASE_TIER,
    GIT_ARCHIVE_TIER,
    HEAD_TIER,
)

SDIST_PACKAGETYPE = "sdist"
WHEEL_PACKAGETYPE = "bdist_wheel"

HTTP_TIMEOUT = float(os.environ.get("AUTOVERSION_HTTP_TIMEOUT", "30"))
USER_AGENT = "uncensorbench-version-check (+https://github.com/wisent-ai/uncensorbench)"

PUBLISHED = "published"
ABSENT = "absent"
UNPROVEN = "unproven"


def fail(message):
    raise SystemExit("baseline.py: " + message)


def fetch(url):
    """Return the response body, or None if no answer arrived at all."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            return exc.read().decode("utf-8", errors="replace")
        except OSError:
            return None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def read_pypi(project):
    """Three states for a PyPI lookup, decided by the answer's content."""
    body = fetch(PYPI_JSON.format(project=project))
    if body is None:
        return UNPROVEN, None
    try:
        document = json.loads(body)
    except ValueError:
        return UNPROVEN, None
    if isinstance(document, dict) and document.get("info", {}).get("name"):
        return PUBLISHED, document
    message = ""
    if isinstance(document, dict):
        message = str(document.get("message", ""))
    if "not found" in message.lower():
        return ABSENT, None
    return UNPROVEN, None


def assert_control():
    """The same lookup, same spelling, against a project PyPI certainly serves."""
    state, _document = read_pypi(CONTROL_PROJECT)
    if state != PUBLISHED:
        fail(
            "the PyPI lookup cannot recognise '"
            + CONTROL_PROJECT
            + "', a project the index definitely serves, so this check is broken"
            " and its verdict about anything else is meaningless"
        )


def resolve_project(root):
    name = surface.project_name(root)
    if not name:
        fail(
            "pyproject.toml declares no [project] name, so there is no subject to"
            " look up; a lookup of an empty name reads as proven absence, which"
            " is exactly the lie this generator must not tell"
        )
    return name


def download(url, destination):
    body_request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(body_request, timeout=HTTP_TIMEOUT) as response:
            destination.write_bytes(response.read())
    except (urllib.error.URLError, OSError) as exc:
        fail("cannot download " + url + ": " + str(exc))


def sole_child(directory):
    entries = [path for path in sorted(directory.iterdir()) if path.is_dir()]
    if not entries:
        fail("the unpacked artifact under " + str(directory) + " has no tree in it")
    return next(iter(entries))


def unpack_sdist(archive, workdir):
    with tarfile.open(archive) as handle:
        handle.extractall(workdir, filter="data")
    return sole_child(workdir)


def unpack_wheel(archive, workdir):
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(workdir)
    return workdir


def declared_version(root):
    return surface.read_pyproject(root).get("project", {}).get("version")


def recover_from_pypi(document, workdir):
    """Recover the newest published version's surface, sdist preferred."""
    version = document["info"]["version"]
    files = document.get("urls", [])
    chosen = None
    for packagetype, tier in ((SDIST_PACKAGETYPE, SDIST_TIER), (WHEEL_PACKAGETYPE, WHEEL_TIER)):
        for entry in files:
            if entry.get("packagetype") == packagetype:
                chosen = (tier, entry)
                break
        if chosen:
            break
    if chosen is None:
        fail(
            "PyPI serves "
            + version
            + " but neither an sdist nor a wheel, so no artifact can be unpacked"
        )
    tier, entry = chosen
    filename = entry["filename"]
    archive = workdir / filename
    download(entry["url"], archive)
    unpacked = workdir / "unpacked"
    unpacked.mkdir()
    if tier == SDIST_TIER:
        tree = unpack_sdist(archive, unpacked)
        inside = declared_version(tree)
        if inside != version:
            fail(
                "the sdist for "
                + version
                + " declares "
                + str(inside)
                + " inside; the artifact disagrees with the release it is filed under"
            )
    else:
        tree = unpack_wheel(archive, unpacked)
    package = surface.resolve_package(tree, None)
    names = surface.collect(tree, package, True)
    marker = tier + ":" + filename
    prose = "recovered from the PyPI release of " + version
    return {"version": version, "source": marker + " " + prose, "surface": names}


def git(root, *arguments):
    result = subprocess.run(
        ("git", *arguments),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def version_key(text):
    parts = text.lstrip("vV").split(".")
    key = []
    for part in parts:
        digits = ""
        for character in part:
            if character.isdigit():
                digits += character
            else:
                break
        if not digits:
            return None
        key.append(int(digits))
    return tuple(key) if key else None


def best_tag(root):
    """The highest tag whose tree really declares the version the tag claims."""
    code, output, _err = git(root, "tag", "--list")
    if code:
        return None
    candidates = []
    for tag in output.split("\n"):
        tag = tag.strip()
        if not tag:
            continue
        key = version_key(tag)
        if key is None:
            continue
        candidates.append((key, tag))
    for key, tag in sorted(candidates, reverse=True):
        with tempfile.TemporaryDirectory() as scratch:
            scratch_path = Path(scratch)
            archive = scratch_path / "tag.tar"
            code, _out, _err = git(root, "archive", "--output", str(archive), tag)
            if code:
                continue
            tree = scratch_path / "tree"
            tree.mkdir()
            with tarfile.open(archive) as handle:
                handle.extractall(tree, filter="data")
            inside = declared_version(tree)
            claimed = tag.lstrip("vV")
            if inside != claimed:
                print(
                    "baseline.py: tag "
                    + tag
                    + " points at a tree declaring "
                    + str(inside)
                    + "; skipping it rather than filing it under "
                    + claimed,
                    file=sys.stderr,
                )
                continue
            package = surface.resolve_package(tree, None)
            names = surface.collect(tree, package, True)
            return {
                "version": claimed,
                "source": GIT_ARCHIVE_TIER
                + ":"
                + tag
                + " reproduced with git archive; nothing is served by PyPI",
                "surface": names,
            }
    return None


def gh_releases_present(root):
    code, output, _err = git(root, "config", "--get", "remote.origin.url")
    if code or not output:
        return False
    trimmed = output.removesuffix(".git")
    prefix, _slash, name = trimmed.rpartition("/")
    _root, _sep, owner = prefix.rpartition("/")
    _host, _colon, owner = owner.rpartition(":")
    if not name or not owner:
        return False
    api = "https://api.github.com/repos/" + owner + "/" + name + "/releases"
    body = fetch(api)
    if body is None:
        fail(
            "the GitHub releases API did not answer, so the absence of a"
            " gh-release tier is unproven"
        )
    try:
        document = json.loads(body)
    except ValueError:
        fail("the GitHub releases API returned no JSON; the gh-release tier is unproven")
    return isinstance(document, list) and bool(document)


def head_baseline(root):
    code, sha, _err = git(root, "rev-parse", "HEAD")
    if code or not sha:
        fail("cannot resolve HEAD, and there is no lower tier than head:")
    package = surface.resolve_package(root, None)
    names = surface.collect(root, package, False)
    return {
        "version": str(declared_version(root)),
        "source": HEAD_TIER
        + ":"
        + sha
        + " the working revision; nothing is served by PyPI and no tag qualifies",
        "surface": names,
    }


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
    parser.add_argument("--root", type=Path, default=here.parent)
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
    print("baseline.py: wrote " + str(target), file=sys.stderr)


if __name__ == "__main__":
    main()
