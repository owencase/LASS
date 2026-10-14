# 1.3.8 설치 백업 수정

1.3.7 설치 프로그램에서 소스·설정 백업이 실패하던 원인을 수정했다. Windows PowerShell 5.1에서는 `System.IO.Compression.FileSystem`만 명시적으로 불러오면 `ZipArchiveMode` 형식을 찾지 못했다. `System.IO.Compression`도 먼저 불러오도록 수정했다.

백업 실패 시 설치는 계속 중단하며, 종료 코드와 `%LOCALAPPDATA%/LASS/logs/installer-backup.log` 경로를 안내한다. 백업 실패를 무시하고 기존 파일을 덮어쓰지 않는다.

검증은 설치 프로그램과 동일한 Windows PowerShell 5.1 프로세스로 수행했다. 한글·공백 경로의 소스와 설정을 ZIP에 보존하고, 접근할 수 없는 백업 경로에서는 실패·로그 기록·원본 보존을 확인했다. 실제 설치된 LASS 소스도 기본 문서/Backups 경로로 백업하여 62개 파일과 ZIP 무결성을 확인했다. 번들 자체 검사도 통과했다.

앱 기능은 1.3.7과 같으며 버전과 설치 백업 처리만 변경했다.
