param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("caseflow", "caseflow_test")]
    [string]$Database = "caseflow"
)

$ErrorActionPreference = "Stop"
$command = @'
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d __DATABASE__ -v checkpoint_password="$CASEFLOW_CHECKPOINT_PASSWORD" -v action_password="$CASEFLOW_ACTION_PASSWORD" -v embedding_password="$CASEFLOW_EMBEDDING_INDEX_PASSWORD" -f /migrations/001_v1c_approvals.sql && psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d __DATABASE__ -v embedding_password="$CASEFLOW_EMBEDDING_INDEX_PASSWORD" -f /migrations/002_v1c_pgvector.sql
'@ -replace "__DATABASE__", $Database
& docker compose exec -T postgres sh -lc $command
if ($LASTEXITCODE -ne 0) {
    throw "V1-C migration failed with exit code $LASTEXITCODE. No container or volume was removed."
}
