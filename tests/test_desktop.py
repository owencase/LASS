import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from src.desktop import DesktopHost, server_command, page


class DesktopTests(unittest.TestCase):
    def test_server_is_local_headless_and_no_file_watcher(self):
        command = server_command(Path('/project with spaces'), 18501)
        self.assertIn('/project with spaces/src/ui/app.py', command)
        self.assertEqual(command[command.index('--server.address') + 1], '127.0.0.1')
        self.assertEqual(command[command.index('--server.headless') + 1], 'true')
        self.assertEqual(command[command.index('--server.fileWatcherType') + 1], 'none')

    def test_window_close_terminates_owned_server(self):
        with tempfile.TemporaryDirectory() as root:
            host = DesktopHost(Path(root))
            host.process = Mock()
            host.process.poll.return_value = None
            host.close()
            host.process.terminate.assert_called_once()
            self.assertTrue(host.stopped.is_set())

    def test_error_page_escapes_details(self):
        self.assertIn('&lt;script&gt;', page('<script>', error=True))
