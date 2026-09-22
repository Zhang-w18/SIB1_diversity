param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RunId = 'remote-001',

    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$configPath = Join-Path $projectRoot 'configs\experiments\plan-005.yaml'
$codebookPath = Join-Path $projectRoot 'outputs\plan-001\codebook_7ghz_8h1v_mainlobe\codebook_weights.npz'
$validationPath = Join-Path $projectRoot (Join-Path 'outputs\plan-005' ("config-validation-" + $RunId))
$outputPath = Join-Path $projectRoot (Join-Path 'outputs\plan-005' (Join-Path $RunId 'full'))

foreach ($requiredPath in @($configPath, $codebookPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required Plan 005 input is missing: $requiredPath"
    }
}

Write-Host 'Checking Python 3.11 runtime...'
& py -3.11 --version
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is unavailable through the py launcher.' }

if (-not $SkipTests) {
    Write-Host 'Running Plan 005 relevant tests...'
    & py -3.11 -m pytest `
        tests/test_cdl_config.py `
        tests/test_cdl.py `
        tests/test_sim.py `
        tests/test_adaptive.py `
        tests/test_plan005.py -q
    if ($LASTEXITCODE -ne 0) { throw 'Plan 005 tests failed.' }
}

Write-Host 'Validating the frozen full-run configuration...'
& py -3.11 -m sib1div.cli validate-config $configPath `
    --output $validationPath --overwrite
if ($LASTEXITCODE -ne 0) { throw 'Plan 005 configuration validation failed.' }

Write-Host "Starting/resuming Plan 005 full BLER simulation: $outputPath"
& py -3.11 -m sib1div.cli simulate $configPath `
    --codebook $codebookPath --output $outputPath
exit $LASTEXITCODE
