import os
import tempfile
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Clear app-specific env while keeping a usable home directory.

    Streamlit's runtime reads `Path.home()` from a side thread on first
    import, and a few tests (notably AppTest) need it to resolve. We keep
    `USERPROFILE`/`HOME` pointing at a temp dir so the AppTest script
    runner can start; everything else is wiped so tests stay hermetic.
    """
    with tempfile.TemporaryDirectory() as home:
        preserved = {"USERPROFILE": home, "HOME": home, "HOMEDRIVE": "C:"}
        with patch.dict(
            os.environ,
            {k: v for k, v in os.environ.items() if k not in preserved} | preserved,
            clear=True,
        ):
            yield
