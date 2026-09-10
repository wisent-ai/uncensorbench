"""Reading modules without importing them: exports, classes and advertised
subcommands, by ``ast`` alone."""

from __future__ import annotations

import ast
import sys

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


