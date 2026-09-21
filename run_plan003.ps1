param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [ValidateSet('estimated', 'ideal', 'wide')]
    [string]$Phase
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$configName = if ($Phase -eq 'wide') { 'plan-003-wide-scan-01.yaml' } else { 'plan-003.yaml' }
$configPath = Join-Path $projectRoot (Join-Path 'configs\experiments' $configName)
$codebookPath = Join-Path $projectRoot 'outputs\plan-001\codebook_7ghz_8h1v_mainlobe\codebook_weights.npz'
$runRoot = Join-Path $projectRoot (Join-Path 'outputs\plan-003' $RunId)
$outputPath = Join-Path $runRoot $Phase

Write-Host "Starting Plan 003 phase: $Phase"
Write-Host "Output: $outputPath"
Write-Host "Re-run with the same RunId and phase to resume an interrupted run."

if ($Phase -eq 'wide') {
    & py -3.11 -m sib1div.cli simulate $configPath --codebook $codebookPath --output $outputPath
} else {
    & py -3.11 -m sib1div.cli simulate $configPath --codebook $codebookPath --output $outputPath --phase $Phase
}
exit $LASTEXITCODE
