#!/usr/bin/env python3
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
import ast
import configparser
import json
import sys

try:
    import tomllib
except ImportError as exc:  # pragma: no cover - depends on the interpreter
    raise SystemExit(
        "surface.py needs an interpreter with tomllib in the standard library"
        " to read pyproject.toml; this one has none (" + str(exc) + ")"
    )

from pathlib import Path

API = "api:"
CLI = "cli:"
SCRIPT = "console-script:"

CONSOLE_SCRIPTS_SECTION = "console_scripts"
SUBPARSER_FACTORY = "add_subparsers"
SUBPARSER_ADD = "add_parser"


def fail(message):
    raise SystemExit("surface.py: " + message)


def parse_module(path, tolerant):
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        if tolerant:
            print(
                "surface.py: skipping unparsable module " + str(path) + ": " + str(exc),
                file=sys.stderr,
            )
            return None
        fail(
            str(path)
            + " does not parse ("
            + str(exc)
            + "); the surface there is unknown, not smaller"
        )
    except OSError as exc:
        fail("cannot read " + str(path) + ": " + str(exc))


def read_pyproject(root):
    manifest = root / "pyproject.toml"
    if not manifest.is_file():
        return {}
    with manifest.open("rb") as handle:
        return tomllib.load(handle)


def project_name(root):
    """The distribution name as the manifest declares it.

    Taken from the manifest, never spelled as a literal at a call site: a
    renamed distribution must make every lookup follow, rather than leave a
    gate happily validating somebody else's project.
    """
    return read_pyproject(root).get("project", {}).get("name")


def entry_points_files(root):
    for pattern in ("*.dist-info/entry_points.txt", "*.egg-info/entry_points.txt"):
        yield from sorted(root.glob(pattern))


def console_scripts(root):
    """Console script name -> ``module:attr`` target.

    ``[project.scripts]`` in ``pyproject.toml`` is authoritative when present.
    A wheel carries no manifest, so fall back to
    ``<dist>.dist-info/entry_points.txt`` under ``[console_scripts]``.
    """
    scripts = read_pyproject(root).get("project", {}).get("scripts")
    if scripts:
        return dict(scripts)
    for path in entry_points_files(root):
        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        if parser.has_section(CONSOLE_SCRIPTS_SECTION):
            return dict(parser.items(CONSOLE_SCRIPTS_SECTION))
    return {}


def resolve_package(root, override):
    if override:
        return override
    name = project_name(root)
    if name:
        return name.replace("-", "_").replace(".", "_")
    for path in sorted(root.glob("*.dist-info/top_level.txt")):
        named = [line.strip() for line in path.read_text(encoding="utf-8").split("\n")]
        named = [line for line in named if line]
        if named:
            return next(iter(named))
    fail(
        "cannot determine the package name under "
        + str(root)
        + "; pass --package explicitly"
    )


def dunder_all(tree, path):
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        value = node.value
        if not isinstance(value, (ast.List, ast.Tuple)):
            fail(str(path) + " defines __all__ as a non-literal; cannot read it")
        names = []
        for element in value.elts:
            if not isinstance(element, ast.Constant) or not isinstance(
                element.value, str
            ):
                fail(str(path) + " has a non-string entry in __all__")
            names.append(element.value)
        return names
    fail(str(path) + " defines no __all__, so the library contract is undeclared")


def relative_imports(tree):
    """Exported name -> the sibling module it is imported from."""
    origin = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            for alias in node.names:
                origin[alias.asname or alias.name] = node.module
    return origin


def class_members(node):
    """Public methods, dataclass fields and enum members declared on a class."""
    members = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not child.name.startswith("_"):
                members.append(child.name)
        elif isinstance(child, ast.AnnAssign):
            target = child.target
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                members.append(target.id)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    members.append(target.id)
    return members


def module_classes(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def cli_commands(path, tolerant):
    """Subcommands the tool's help advertises: ``add_parser`` calls with ``help=``.

    The receiver must be a name bound to an ``add_subparsers()`` result, so an
    ``add_parser`` on some unrelated object is not mistaken for a command.
    """
    if not path.is_file():
        if tolerant:
            print("surface.py: missing cli module " + str(path), file=sys.stderr)
            return []
        fail("the console script points at a missing module " + str(path))
    tree = parse_module(path, tolerant)
    if tree is None:
        return []
    holders = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr == SUBPARSER_FACTORY:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    holders.add(target.id)
    commands = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != SUBPARSER_ADD:
            continue
        if not isinstance(func.value, ast.Name) or func.value.id not in holders:
            continue
        if not node.args:
            continue
        first = next(iter(node.args))
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        if any(keyword.arg == "help" for keyword in node.keywords):
            commands.append(first.value)
    return commands


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
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Print the public surface of the uncensorbench distribution."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=here.parent,
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


if __name__ == "__main__":
    main()
