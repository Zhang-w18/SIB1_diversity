param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RunId
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$configPath = Join-Path $projectRoot 'configs\experiments\plan-001.yaml'
$codebookPath = Join-Path $projectRoot 'outputs\plan-001\codebook_7ghz_8h1v_mainlobe\codebook_weights.npz'
$outputPath = Join-Path $projectRoot (Join-Path 'outputs\plan-001' $RunId)

Write-Host "Starting Plan 001"
Write-Host "Output: $outputPath"
Write-Host "Re-run this same command with the same RunId to resume an interrupted run."

& py -3.11 -m sib1div.cli simulate $configPath --codebook $codebookPath --output $outputPath
exit $LASTEXITCODE
