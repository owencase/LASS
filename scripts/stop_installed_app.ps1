param([Parameter(Mandatory=$true)][string]$AppExe)
$ErrorActionPreference = 'Stop'
$ownedExe = [IO.Path]::GetFullPath($AppExe)
foreach ($candidate in Get-Process -Name LASS -ErrorAction SilentlyContinue) {
    try { $candidatePath = $candidate.Path } catch { continue }
    if ($candidatePath -and ([IO.Path]::GetFullPath($candidatePath) -eq $ownedExe)) {
        & "$env:SystemRoot/System32/taskkill.exe" /PID $candidate.Id /T /F | Out-Null
    }
}
exit 0
