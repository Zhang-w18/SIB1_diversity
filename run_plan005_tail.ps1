param(
    [ValidateSet('prepare-prescan', 'prescan', 'prepare-fast-prescan', 'fast-prescan', 'prepare-tail', 'tail', 'auto-tail', 'merge', 'rsrp')]
    [string]$Phase = 'prepare-prescan',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RunId = 'run-002'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$codebook = Join-Path $projectRoot 'outputs\plan-001\codebook_7ghz_8h1v_mainlobe\codebook_weights.npz'
$baseConfig = Join-Path $projectRoot 'configs\experiments\plan-005.yaml'
$prescanConfig = Join-Path $projectRoot 'configs\experiments\plan-005-tail-prescan.yaml'
$fastPrescanConfig = Join-Path $projectRoot 'configs\experiments\plan-005-tail-fast-prescan.yaml'
$configDir = Join-Path $projectRoot 'configs\experiments\plan-005-tail-generated'
$runRoot = Join-Path $projectRoot (Join-Path 'outputs\plan-005' $RunId)

if ($Phase -eq 'prepare-prescan') {
    & py -3.11 -m sib1div.analysis.plan005_tail prepare-prescan --base-config $baseConfig --output-config $prescanConfig
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & py -3.11 -m sib1div.cli validate-config $prescanConfig --output (Join-Path $runRoot 'config-validation-prescan')
} elseif ($Phase -eq 'prescan') {
    & py -3.11 -m sib1div.cli simulate $prescanConfig --codebook $codebook --output (Join-Path $runRoot 'prescan-1pct')
} elseif ($Phase -eq 'prepare-fast-prescan') {
    & py -3.11 -m sib1div.analysis.plan005_tail prepare-fast-prescan --base-config $baseConfig --output-config $fastPrescanConfig
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & py -3.11 -m sib1div.cli validate-config $fastPrescanConfig --output (Join-Path $runRoot 'config-validation-fast-prescan')
} elseif ($Phase -eq 'fast-prescan') {
    & py -3.11 -m sib1div.cli simulate $fastPrescanConfig --codebook $codebook --output (Join-Path $runRoot 'prescan-1pct-fast')
} elseif ($Phase -eq 'prepare-tail') {
    & py -3.11 -m sib1div.analysis.plan005_tail prepare-tail `
        --prescan-csv (Join-Path $runRoot 'prescan-1pct-fast\bler.csv') `
        --coarse-prescan-csv (Join-Path $projectRoot 'outputs\plan-005\run-001\prescan\bler.csv') `
        --existing-csv (Join-Path $projectRoot 'outputs\plan-005\run-001\full\bler.csv') `
        --base-config $baseConfig --output-dir $configDir `
        --selection-json (Join-Path $runRoot 'selection-1pct.json')
} elseif ($Phase -eq 'tail') {
    $selection = Get-Content -Raw -Encoding UTF8 (Join-Path $runRoot 'selection-1pct.json') | ConvertFrom-Json
    foreach ($point in $selection.points) {
        $snrToken = if ([double]$point.snr_db -lt 0) { 'm' } else { 'p' }
        $snrToken += ([Math]::Abs([double]$point.snr_db).ToString('0.0', [Globalization.CultureInfo]::InvariantCulture)).Replace('.', 'p')
        $pointOutput = Join-Path $runRoot "tail\snr-$snrToken"
        if (Test-Path -LiteralPath (Join-Path $pointOutput 'run_metadata.json')) {
            Write-Host "Skipping completed tail point $($point.snr_db) dB"
            continue
        }
        & py -3.11 -m sib1div.cli simulate $point.config --codebook $codebook --output $pointOutput
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} elseif ($Phase -eq 'auto-tail') {
    & py -3.11 -m sib1div.analysis.plan005_tail prepare-fast-prescan --base-config $baseConfig --output-config $fastPrescanConfig
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $fastOutput = Join-Path $runRoot 'prescan-1pct-fast'
    if (-not (Test-Path -LiteralPath (Join-Path $fastOutput 'run_metadata.json'))) {
        & py -3.11 -m sib1div.cli simulate $fastPrescanConfig --codebook $codebook --output $fastOutput
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        Write-Host 'Skipping completed fast prescan'
    }
    & py -3.11 -m sib1div.analysis.plan005_tail prepare-tail `
        --prescan-csv (Join-Path $fastOutput 'bler.csv') `
        --coarse-prescan-csv (Join-Path $projectRoot 'outputs\plan-005\run-001\prescan\bler.csv') `
        --existing-csv (Join-Path $projectRoot 'outputs\plan-005\run-001\full\bler.csv') `
        --base-config $baseConfig --output-dir $configDir `
        --selection-json (Join-Path $runRoot 'selection-1pct.json')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $selection = Get-Content -Raw -Encoding UTF8 (Join-Path $runRoot 'selection-1pct.json') | ConvertFrom-Json
    foreach ($point in $selection.points) {
        $snrToken = if ([double]$point.snr_db -lt 0) { 'm' } else { 'p' }
        $snrToken += ([Math]::Abs([double]$point.snr_db).ToString('0.0', [Globalization.CultureInfo]::InvariantCulture)).Replace('.', 'p')
        $pointOutput = Join-Path $runRoot "tail\snr-$snrToken"
        if (Test-Path -LiteralPath (Join-Path $pointOutput 'run_metadata.json')) {
            Write-Host "Skipping completed tail point $($point.snr_db) dB"
            continue
        }
        & py -3.11 -m sib1div.cli simulate $point.config --codebook $codebook --output $pointOutput
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} elseif ($Phase -eq 'merge') {
    & py -3.11 -m sib1div.analysis.plan005_tail merge `
        --existing-dir (Join-Path $projectRoot 'outputs\plan-005\run-001\full') `
        --tail-root (Join-Path $runRoot 'tail') --output (Join-Path $runRoot 'combined')
} else {
    & py -3.11 -m sib1div.cli fixed-cdl-rsrp-cdf $baseConfig --codebook $codebook `
        --output (Join-Path $runRoot 'rsrp-cdf') --drop-start 2000 --drops 10000
}
exit $LASTEXITCODE
