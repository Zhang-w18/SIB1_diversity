param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RunId = 'run-004',

    [ValidateSet('threshold', 'core', 'all')]
    [string]$Batch = 'all'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$codebookPath = Join-Path $projectRoot 'outputs\plan-001\codebook_7ghz_8h1v_mainlobe\codebook_weights.npz'
$runRoot = Join-Path $projectRoot (Join-Path 'outputs\plan-003' $RunId)

function Invoke-Plan003Batch {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$ConfigName
    )

    $configPath = Join-Path $projectRoot (Join-Path 'configs\experiments' $ConfigName)
    $outputPath = Join-Path $runRoot $Name
    Write-Host "Starting Plan 003 append batch: $Name"
    Write-Host "Output: $outputPath"
    Write-Host 'Progress lines include overall percentage, throughput, elapsed time, remaining time, and UTC finish estimate.'
    & py -3.11 -m sib1div.cli simulate $configPath --codebook $codebookPath --output $outputPath
    if ($LASTEXITCODE -ne 0) {
        throw "Plan 003 append batch '$Name' failed with exit code $LASTEXITCODE"
    }
}

if ($Batch -in @('threshold', 'all')) {
    Invoke-Plan003Batch -Name 'threshold' -ConfigName 'plan-003-threshold-scan-01.yaml'
}
if ($Batch -in @('core', 'all')) {
    Invoke-Plan003Batch -Name 'core' -ConfigName 'plan-003-smooth-core-01.yaml'
}
