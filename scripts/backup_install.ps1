param([Parameter(Mandatory=$true)][string]$SourceRoot, [string]$LogPath)
$ErrorActionPreference = 'Stop'
try {
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$taskSource = [IO.Path]::GetFullPath((Join-Path $SourceRoot '_internal'))
if (-not (Test-Path -LiteralPath (Join-Path $taskSource 'src'))) { exit 0 }
$taskDestination = $env:LASS_BACKUP_DIR
if (-not $taskDestination) { $taskDestination = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Backups' }
[IO.Directory]::CreateDirectory($taskDestination) | Out-Null
$taskArchive = Join-Path $taskDestination ('LASS-before-update-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff') + '.zip')
$taskZip = [IO.Compression.ZipFile]::Open($taskArchive, [IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($taskFolder in @('src','prompts','.streamlit')) {
        $taskPath = Join-Path $taskSource $taskFolder
        if (Test-Path -LiteralPath $taskPath) {
            Get-ChildItem -LiteralPath $taskPath -Force -Recurse -File | Where-Object {
                -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -and $_.FullName -notmatch '__pycache__'
            } | ForEach-Object {
                $taskName = $_.FullName.Substring($taskSource.Length + 1).Replace('\','/')
                [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($taskZip, $_.FullName, $taskName) | Out-Null
            }
        }
    }
    foreach ($taskName in @('.env','.env.example','pyproject.toml')) {
        $taskPath = Join-Path $SourceRoot $taskName
        if (Test-Path -LiteralPath $taskPath) {
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($taskZip, $taskPath, $taskName) | Out-Null
        }
    }
} finally { $taskZip.Dispose() }
Write-Output $taskArchive
if ($LogPath) {
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($LogPath)) | Out-Null
    "OK: $taskArchive" | Out-File -LiteralPath $LogPath -Encoding utf8
}
exit 0
} catch {
    if ($LogPath) {
        try {
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($LogPath)) | Out-Null
            $_.Exception.ToString() | Out-File -LiteralPath $LogPath -Encoding utf8
        } catch { }
    }
    Write-Error $_ -ErrorAction Continue
    exit 1
}
