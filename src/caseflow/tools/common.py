from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError


class ToolFailure(Exception):
    def __init__(self, code: str, public_message: str):
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


def translate_database_error(error: SQLAlchemyError) -> ToolFailure:
    detail = str(error).lower()
    if isinstance(error, OperationalError) and (
        "statement timeout" in detail or "canceling statement" in detail
    ):
        return ToolFailure("DATABASE_TIMEOUT", "The read-only database query exceeded its time limit.")
    if isinstance(error, DBAPIError) and not error.connection_invalidated:
        return ToolFailure("DATABASE_QUERY_FAILED", "The read-only database query failed.")
    return ToolFailure("DATABASE_UNAVAILABLE", "The PostgreSQL database is unavailable.")
