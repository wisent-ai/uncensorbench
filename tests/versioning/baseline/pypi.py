"""The PyPI tiers: what the index serves for this project, and the surface
recovered from the artifact it serves."""

from __future__ import annotations

import json
import tarfile
import urllib.error
import urllib.request
import zipfile

from constants import (
    ABSENT, CONTROL_PROJECT, PUBLISHED, PYPI_JSON, SDIST_PACKAGETYPE, SDIST_TIER, UNPROVEN,
    USER_AGENT, WHEEL_PACKAGETYPE, WHEEL_TIER,
)
from uncensorbench import surface  # noqa: E402  (the entry point puts the checkout on sys.path)


def fail(message):
    raise SystemExit("baseline: " + message)


def fetch(url):
    """Return the response body, or None if no answer arrived at all."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response:
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
        with urllib.request.urlopen(body_request) as response:
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


