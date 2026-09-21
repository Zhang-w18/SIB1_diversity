param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RunId = 'run-001',

    [ValidateSet('threshold-integer-fill', 'threshold-half-grid', 'tail-balance', 'low-bler-refine', 'all')]
    [string]$Batch = 'all'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$codebookPath = Join-Path $projectRoot 'outputs\plan-001\codebook_7ghz_8h1v_mainlobe\codebook_weights.npz'
$runRoot = Join-Path $projectRoot (Join-Path 'outputs\plan-004' $RunId)

function Invoke-Plan004Batch {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$ConfigName
    )
    $configPath = Join-Path $projectRoot (Join-Path 'configs\experiments' $ConfigName)
    $outputPath = Join-Path $runRoot $Name
    Write-Host "Starting Plan 004 batch: $Name"
    Write-Host "Output: $outputPath"
    & py -3.11 -m sib1div.cli simulate $configPath --codebook $codebookPath --output $outputPath
    if ($LASTEXITCODE -ne 0) {
        throw "Plan 004 batch '$Name' failed with exit code $LASTEXITCODE"
    }
}

$batches = @(
    [pscustomobject]@{ Name = 'threshold-integer-fill'; Config = 'plan-004-threshold-integer-fill.yaml' }
    [pscustomobject]@{ Name = 'threshold-half-grid'; Config = 'plan-004-threshold-half-grid.yaml' }
    [pscustomobject]@{ Name = 'tail-balance'; Config = 'plan-004-tail-balance.yaml' }
    [pscustomobject]@{ Name = 'low-bler-refine'; Config = 'plan-004-low-bler-refine.yaml' }
)
foreach ($item in $batches) {
    if ($Batch -eq 'all' -or $Batch -eq $item.Name) {
        Invoke-Plan004Batch -Name $item.Name -ConfigName $item.Config
    }
}
