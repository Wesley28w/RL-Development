[CmdletBinding()]
param(
    [string] $Ablations = "Uniform,AbsoluteDifficulty,Fixed5050Blend,ArgmaxOnly,SoftmaxOnly,NoEMA,SamplingRatio05,Delay15,BufferSize1,VelocityWorld,SimplifiedSubtasks",
    [int[]] $Seeds = (42..53),
    [switch] $AllowMissing,
    [switch] $Overwrite,
    [switch] $DryRun,
    [string] $PythonExe
)
$ErrorActionPreference = "Stop"
$ablationList = @($Ablations -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
& (Join-Path $PSScriptRoot "evaluate_campaign.ps1") -Tasks Cabinet -Methods $ablationList -Seeds $Seeds -AllowMissing:$AllowMissing -Overwrite:$Overwrite -DryRun:$DryRun -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
