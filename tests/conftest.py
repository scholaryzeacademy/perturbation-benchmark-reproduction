import pytest


@pytest.fixture
def data_dir() -> str:
    """Fixed (not per-test-random) download directory for integration tests,
    so GEARS's PertData downloads land in ./data -- matching .gitignore,
    .dockerignore, and the CI cache key in .github/workflows/ci.yml. Using
    pytest's default tmp_path here would defeat that cache and re-download
    the dataset on every run."""
    return "data"
