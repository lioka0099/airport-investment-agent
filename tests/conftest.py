import pytest

from airport_agent.data.store import Store
from tests.fixtures import build_snapshot


@pytest.fixture(scope="session")
def snapshot_dir(tmp_path_factory):
    return build_snapshot(tmp_path_factory.mktemp("snapshot"))


@pytest.fixture(scope="session")
def store(snapshot_dir):
    return Store(snapshot_dir)
