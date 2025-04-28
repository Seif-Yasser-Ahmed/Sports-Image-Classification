<#
.SYNOPSIS
  Download & unzip a Kaggle dataset using the Kaggle CLI.
.PARAMETER Slug
  The Kaggle dataset slug, e.g. "sidharkal/sports-image-classification".
.EXAMPLE
  .\download_dataset.ps1 -Slug sidharkal/sports-image-classification
#>

param (
  [Parameter(Mandatory = $true)]
  [string]$Slug
)

# turn user/dataset → user-dataset
$Folder    = $Slug -replace '/', '-'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$DestDir   = Join-Path $ScriptDir $Folder

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: kaggle CLI not found. Install via 'pip install kaggle' and configure your ~/.kaggle/kaggle.json"
    exit 1
}

# Create the destination folder
New-Item -ItemType Directory -Path $DestDir -Force | Out-Null

# Simplified, plain-ASCII status message
Write-Host "Downloading dataset '$Slug' to '$DestDir' ..."

# Run the download
kaggle datasets download -d $Slug --unzip -p $DestDir

Write-Host "Done. Dataset available in $DestDir"
