param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("caseflow", "caseflow_test")]
    [string]$Database = "caseflow_test"
)

$ErrorActionPreference = "Stop"
$dockerCommand = Get-Command docker.exe -ErrorAction SilentlyContinue
if ($dockerCommand) {
    $docker = $dockerCommand.Source
} else {
    $docker = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"
}
if (-not (Test-Path -LiteralPath $docker)) {
    throw "Docker CLI was not found. Start Docker Desktop and retry; no database was changed."
}

$container = & $docker compose ps --status running --services postgres
if ($LASTEXITCODE -ne 0 -or $container -notcontains "postgres") {
    throw "The Compose PostgreSQL service is not running; no database was changed."
}

$command = "psql -v ON_ERROR_STOP=1 -U `"`$POSTGRES_USER`" -d $Database -f /migrations/003_v1d_pending_draft_revision.sql"
& $docker compose exec -T postgres sh -lc $command
if ($LASTEXITCODE -ne 0) {
    throw "V1-D approval migration failed with exit code $LASTEXITCODE. No container or volume was removed."
}
Write-Output "V1-D approval migration applied to $Database."
