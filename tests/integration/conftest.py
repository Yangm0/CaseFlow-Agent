import pytest

from caseflow.config import get_settings
from caseflow.data.seed import seed_database
from caseflow.db import make_engine


@pytest.fixture(scope="session")
def postgres_engines():
    settings = get_settings()
    app_url = settings.test_database_url
    seed_url = settings.test_seed_database_url
    if not app_url or not seed_url:
        pytest.skip("TEST_DATABASE_URL and TEST_SEED_DATABASE_URL are not configured")
    app_engine = make_engine(app_url)
    seed_engine = make_engine(seed_url)
    app_database = str(app_engine.url.database or "")
    seed_database_name = str(seed_engine.url.database or "")
    if app_database != "caseflow_test" or seed_database_name != "caseflow_test":
        app_engine.dispose()
        seed_engine.dispose()
        pytest.fail("Integration tests only allow the isolated caseflow_test database.")
    try:
        seed_database(seed_engine)
        with app_engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
    except Exception:
        app_engine.dispose()
        seed_engine.dispose()
        pytest.fail("The isolated PostgreSQL test database is unavailable or not initialized.")
    yield app_engine, seed_engine
    app_engine.dispose()
    seed_engine.dispose()
