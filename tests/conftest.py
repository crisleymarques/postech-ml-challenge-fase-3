import pytest
from sources.config import setup_dirs


@pytest.fixture(scope="session", autouse=True)
def _setup_project_dirs():
    setup_dirs()
