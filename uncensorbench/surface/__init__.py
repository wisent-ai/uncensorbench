"""Print the public surface of the ``uncensorbench`` distribution.

Why this set is the contract
----------------------------
``uncensorbench`` is published to PyPI as a single distribution that is three
products at once, and a consumer can hold any of the three:

* a **library** -- ``from uncensorbench import UncensorBench``;
* a **console script** -- ``uncensorbench`` is declared in ``[project.scripts]``,
  so a rename breaks a shell script that ran yesterday;
* a **command-line tool** -- the subcommands its ``--help`` advertises.

So the surface is the union of four namespaced families:

``api:<Name>``            an entry of ``uncensorbench.__all__``.
``api:<Class>.<member>``  a public member of an exported class: a method, a
                          dataclass field, or an enum member. Counting only the
                          class names would call ``UncensorBench.evaluate``
                          internal, and deleting it is plainly a removal --
                          this is the "include a set whose removal your surface
                          would otherwise call internal" rule from ADOPTING.md.
``cli:<command>``         a subcommand registered with a ``help=`` text, i.e.
                          one the tool's own help *advertises*. A subparser
                          added without ``help=`` dispatches but is unlisted,
                          and unlisted means private.
``console-script:<name>`` a console script name from the packaging metadata.

Deliberately excluded: option flags (argparse detail that churns), the prompt
and topic JSON payloads (data the benchmark is expected to grow), and anything
underscore-prefixed.

Everything is read statically with ``ast``. The package is never imported: a
release decision must not require a machine that can import ``torch``, and the
same reader has to run against an unpacked published sdist or wheel, which is
how ``released-surface.json`` is recovered rather than assumed.

A module that fails to parse is a hard error. Skipping it would report a
shorter surface, and a shorter surface reads as removed capability -- a false
``breaking`` verdict for an unrelated syntax error. ``--tolerant`` downgrades
that to a warning and is meant only for recovering an already-published
artifact; it names every module it skipped on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .modules import (
    class_members, cli_commands, dunder_all, fail, module_classes, parse_module, relative_imports,
)
from .packaging import console_scripts, resolve_package
from .packaging import project_name as project_name, read_pyproject as read_pyproject

API = "api:"
CLI = "cli:"
SCRIPT = "console-script:"


def collect(root, package, tolerant):
    package_dir = root / package
    if not package_dir.is_dir():
        fail("no package directory at " + str(package_dir))
    init = package_dir / "__init__.py"
    if not init.is_file():
        fail("no " + str(init))

    tree = parse_module(init, tolerant)
    if tree is None:
        fail(str(init) + " does not parse; the whole contract is unknown")

    exported = dunder_all(tree, init)
    origin = relative_imports(tree)

    parsed = {}
    for name in exported:
        module = origin.get(name)
        if module is None or module in parsed:
            continue
        module_path = package_dir / (module.replace(".", "/") + ".py")
        if not module_path.is_file():
            if tolerant:
                print("surface.py: missing module " + str(module_path), file=sys.stderr)
                parsed[module] = None
                continue
            fail("__all__ names " + name + " from a missing module " + str(module_path))
        parsed[module] = parse_module(module_path, tolerant)

    surface = set()
    for name in exported:
        surface.add(API + name)
        tree_for = parsed.get(origin.get(name))
        if tree_for is None:
            continue
        node = module_classes(tree_for).get(name)
        if node is None:
            continue
        for member in class_members(node):
            surface.add(API + name + "." + member)

    for script, target in console_scripts(root).items():
        surface.add(SCRIPT + script)
        module, separator, _attribute = target.partition(":")
        if not separator:
            continue
        head, _dot, tail = module.strip().partition(".")
        if head != package or not tail:
            continue
        for command in cli_commands(
            package_dir / (tail.replace(".", "/") + ".py"), tolerant
        ):
            surface.add(CLI + command)

    return sorted(surface)


def main():
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description="Print the public surface of the uncensorbench distribution."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=here.parents[2],
        help="tree to read: the repository, or an unpacked sdist or wheel",
    )
    parser.add_argument("--package", help="import package name; inferred by default")
    parser.add_argument(
        "--tolerant",
        action="store_true",
        help="warn instead of failing on an unparsable module (recovery only)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        fail("no such tree: " + str(root))
    package = resolve_package(root, args.package)
    print(json.dumps({"surface": collect(root, package, args.tolerant)}, indent="  "))


