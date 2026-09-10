"""What the packaging metadata declares: the project, its console scripts and
the package the surface is read from."""

from __future__ import annotations

import configparser

try:
    import tomllib
except ImportError as exc:  # pragma: no cover - depends on the interpreter
    raise SystemExit(
        "uncensorbench.surface needs an interpreter with tomllib in the standard library"
        " to read pyproject.toml; this one has none (" + str(exc) + ")"
    )

from .modules import fail

CONSOLE_SCRIPTS_SECTION = "console_scripts"


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


