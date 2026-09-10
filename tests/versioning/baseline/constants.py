"""The tiers a baseline can come from, and the PyPI vocabulary the generator reads."""

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

USER_AGENT = "uncensorbench-version-check (+https://github.com/wisent-ai/uncensorbench)"

PUBLISHED = "published"
ABSENT = "absent"
UNPROVEN = "unproven"

