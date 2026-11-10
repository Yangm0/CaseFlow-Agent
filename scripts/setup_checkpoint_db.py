from __future__ import annotations

import argparse
import os
import sys

from caseflow.checkpoints.runtime import _driver_uri


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize the explicitly selected checkpoint DB."
    )
    parser.add_argument("--test", action="store_true", help="Use TEST_CHECKPOINT_DATABASE_URL.")
    args = parser.parse_args()
    variable = "TEST_CHECKPOINT_DATABASE_URL" if args.test else "CHECKPOINT_DATABASE_URL"
    database_url = os.environ.get(variable)
    if not database_url:
        print(f"{variable} is required; checkpoint setup was not run.")
        return 2

    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(_driver_uri(database_url)) as checkpointer:
        checkpointer.setup()
    print("LangGraph checkpoint tables created in the configured checkpoint schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
