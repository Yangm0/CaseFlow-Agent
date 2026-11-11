from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy.engine import make_url

from caseflow.checkpoints.runtime import _driver_uri
from caseflow.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize the explicitly selected checkpoint DB."
    )
    parser.add_argument("--test", action="store_true", help="Use TEST_CHECKPOINT_DATABASE_URL.")
    args = parser.parse_args()
    settings = get_settings()
    variable = "TEST_CHECKPOINT_DATABASE_URL" if args.test else "CHECKPOINT_DATABASE_URL"
    database_url = (
        settings.test_checkpoint_database_url if args.test else settings.checkpoint_database_url
    )
    if not database_url:
        print(f"{variable} is required in the environment or project .env; setup was not run.")
        return 2
    expected_database = "caseflow_test" if args.test else "caseflow"
    try:
        selected_database = make_url(database_url).database
    except Exception:
        print(f"{variable} is not a valid PostgreSQL URL; checkpoint setup was not run.")
        return 2
    if selected_database != expected_database:
        print(
            f"{variable} must target {expected_database}; checkpoint setup was not run."
        )
        return 2

    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(_driver_uri(database_url)) as checkpointer:
        checkpointer.setup()
    print("LangGraph checkpoint tables created in the configured checkpoint schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
