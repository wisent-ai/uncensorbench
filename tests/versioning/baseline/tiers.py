"""The git tiers: the best tag the repository carries, a GitHub release, or
the working tree itself when nothing was ever published."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from constants import GIT_ARCHIVE_TIER, HEAD_TIER
from pypi import declared_version, fail, fetch
from uncensorbench import surface  # noqa: E402  (the entry point puts the checkout on sys.path)


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
                    "baseline: tag "
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


