param(
    [ValidateSet('diagnostic', 'prescan', 'prepare', 'full')]
    [string]$Phase = 'diagnostic',

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RunId = 'run-001'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$codebookPath = Join-Path $projectRoot 'outputs\plan-001\codebook_7ghz_8h1v_mainlobe\codebook_weights.npz'
$runRoot = Join-Path $projectRoot (Join-Path 'outputs\plan-005' $RunId)

if ($Phase -eq 'diagnostic') {
    $configPath = Join-Path $projectRoot 'configs\experiments\plan-005-prescan.yaml'
    $outputPath = Join-Path $runRoot 'beam-rsrp'
    & py -3.11 -m sib1div.cli diagnose-fixed-cdl $configPath --codebook $codebookPath --output $outputPath
} elseif ($Phase -eq 'prescan') {
    $configPath = Join-Path $projectRoot 'configs\experiments\plan-005-prescan.yaml'
    $outputPath = Join-Path $runRoot 'prescan'
    & py -3.11 -m sib1div.cli simulate $configPath --codebook $codebookPath --output $outputPath
} elseif ($Phase -eq 'prepare') {
    $prescanCsv = Join-Path $runRoot 'prescan\bler.csv'
    $prescanConfig = Join-Path $projectRoot 'configs\experiments\plan-005-prescan.yaml'
    $fullConfig = Join-Path $projectRoot 'configs\experiments\plan-005.yaml'
    $selectionJson = Join-Path $runRoot 'snr-selection.json'
    & py -3.11 -m sib1div.analysis.plan005 --prescan-csv $prescanCsv --prescan-config $prescanConfig --output-config $fullConfig --selection-json $selectionJson
} else {
    $configPath = Join-Path $projectRoot 'configs\experiments\plan-005.yaml'
    if (-not (Test-Path -LiteralPath $configPath)) {
        throw 'Full-run config is frozen only after the prescan: configs/experiments/plan-005.yaml'
    }
    $outputPath = Join-Path $runRoot 'full'
    & py -3.11 -m sib1div.cli simulate $configPath --codebook $codebookPath --output $outputPath
}
exit $LASTEXITCODE
