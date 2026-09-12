import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sources.config import setup_dirs


@pytest.fixture(scope="session", autouse=True)
def _setup_project_dirs():
    setup_dirs()
