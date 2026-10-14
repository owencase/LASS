"""Exercise the exact Windows PowerShell runtime launched by Inno Setup."""
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


@unittest.skipUnless(os.name == "nt", "Windows installer")
class InstallerBackupTests(unittest.TestCase):
    def test_windows_powershell_backs_up_unicode_paths_and_reports_failure(self):
        shell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
        script = Path(__file__).resolve().parents[1] / 'scripts/backup_install.ps1'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / '기존 설치 폴더'
            (source / '_internal/src').mkdir(parents=True)
            (source / '_internal/src/main.py').write_text('original source', encoding='utf-8')
            (source / '.env').write_text('SETTING=original', encoding='utf-8')
            destination = root / '백업 폴더'
            log = root / 'logs/backup.log'
            command = [str(shell), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                       '-File', str(script), '-SourceRoot', str(source), '-LogPath', str(log)]
            environment = dict(os.environ, LASS_BACKUP_DIR=str(destination))
            result = subprocess.run(command, env=environment, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            with zipfile.ZipFile(next(destination.glob('*.zip'))) as archive:
                self.assertEqual(archive.read('src/main.py'), b'original source')
                self.assertEqual(archive.read('.env'), b'SETTING=original')
                self.assertIsNone(archive.testzip())
            self.assertIn('OK:', log.read_text(encoding='utf-8-sig'))
            blocked = root / 'blocked'
            blocked.write_text('file instead of directory')
            environment['LASS_BACKUP_DIR'] = str(blocked)
            result = subprocess.run(command, env=environment, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('OK:', log.read_text(encoding='utf-8-sig'))
            self.assertEqual((source / '_internal/src/main.py').read_text(), 'original source')
