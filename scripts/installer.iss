#define MyAppName "LASS IQ4 Moonshine"
#ifndef MyAppVersion
  #define MyAppVersion "2.1.1"
#endif
#ifndef MyDataRoot
  #define MyDataRoot "{localappdata}\LASS-IQ4-Moonshine"
#endif
[Setup]
AppId={{F3B8D2EA-B2BC-4B81-AFC2-80EC97100A56}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=LASS
LicenseFile=..\src\assets\MOONSHINE-LICENSE.txt
DefaultDirName={localappdata}\Programs\LASS-IQ4-Moonshine
DefaultGroupName=LASS IQ4 Moonshine
PrivilegesRequired=lowest
MinVersion=10.0.18362
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=LASS-Setup-{#MyAppVersion}
SetupIconFile=..\src\assets\lass.ico
UninstallDisplayIcon={app}\LASS.exe
Compression=lzma2/fast
SolidCompression=yes
LZMANumBlockThreads=2
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
UninstallDisplayName=LASS IQ4 Moonshine
; v1.3.0 logged destructive UninstallDelete entries; appending retains them.
; All installed application files are enumerated by [Files] on every release.
UninstallLogMode=overwrite
[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
[Files]
Source: "backup_install.ps1"; Flags: dontcopy
Source: "stop_installed_app.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\LASS\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\LASS IQ4 Moonshine"; Filename: "{app}\LASS.exe"
Name: "{autodesktop}\LASS IQ4 Moonshine"; Filename: "{app}\LASS.exe"
[Run]
Filename: "{app}\LASS.exe"; Description: "LASS 실행"; Flags: nowait postinstall skipifsilent
; Upgrades preserve data; an explicit uninstall removes this application's data.
[UninstallRun]
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File ""{app}\stop_installed_app.ps1"" -AppExe ""{app}\LASS.exe"""; Flags: runhidden; RunOnceId: "StopLASSMoonshine"
[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{#MyDataRoot}"
Type: filesandordirs; Name: "{app}"; Check: DedicatedInstallDirectory

[Code]
function DedicatedInstallDirectory: Boolean;
begin
  { Only delete an entire directory bearing this application's dedicated name. }
  Result := CompareText(ExtractFileName(ExpandConstant('{app}')), 'LASS-IQ4-Moonshine') = 0;
end;

function InitializeUninstall: Boolean;
begin
  Result := True;
  if not UninstallSilent then
    Result := MsgBox('프로그램과 함께 노트, 첨부파일, 모델, 캐시, 로그를 모두 삭제합니다. 계속하시겠습니까?', mbConfirmation, MB_YESNO) = IDYES;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  Result := '';
  if not FileExists(ExpandConstant('{app}\_internal\src\__init__.py')) then Exit;
  ExtractTemporaryFile('backup_install.ps1');
  if not Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
    '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\backup_install.ps1') +
    '" -SourceRoot "' + ExpandConstant('{app}') + '" -LogPath "' + ExpandConstant('{localappdata}\LASS-IQ4-Moonshine\logs\installer-backup.log') + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
    Result := '기존 소스·설정 백업에 실패하여 업데이트를 중단했습니다. 오류 코드: ' + IntToStr(ResultCode) + #13#10 +
      '상세 기록: ' + ExpandConstant('{localappdata}\LASS-IQ4-Moonshine\logs\installer-backup.log');
end;
