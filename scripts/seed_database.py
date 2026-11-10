from caseflow.config import get_settings
from caseflow.data.seed import seed_database
from caseflow.db import make_engine


def main() -> None:
    settings = get_settings()
    if not settings.seed_database_url:
        raise SystemExit("SEED_DATABASE_URL is required; the application DATABASE_URL is read-only.")
    seed_engine = make_engine(settings.seed_database_url)
    try:
        counts = seed_database(seed_engine)
    finally:
        seed_engine.dispose()
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
