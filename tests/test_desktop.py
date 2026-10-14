import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from src.desktop import DesktopHost, server_command, page, serve, prepare_stdio
import sys


class DesktopTests(unittest.TestCase):
    def test_workspace_starts_without_waiting_for_model_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            host = DesktopHost(Path(directory))
            host.window = Mock()
            host.stopped = Mock()
            host.stopped.wait.side_effect = [False, True]
            host._spawn = Mock()
            host._spawn.return_value.poll.return_value = None
            with patch.dict('os.environ'), patch('src.desktop_picker.FilePickerServer') as picker, patch('src.desktop.urlopen') as request:
                picker.return_value.url = 'http://127.0.0.1:12345'
                request.return_value.__enter__.return_value.status = 200
                host.start_server()
            host.window.load_url.assert_called_once()

    def test_windowed_exe_streams_support_server_logging(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, 'stdout', None), patch.object(sys, 'stderr', None):
            prepare_stdio(Path(directory), 'server')
            stream = sys.stdout
            try:
                self.assertFalse(stream.isatty())
                self.assertIs(sys.stderr, stream)
                stream.write('server startup\n')
                stream.flush()
                self.assertIn('server startup', (Path(directory) / 'data/logs/stdio-server.log').read_text(encoding='utf-8'))
            finally:
                stream.close()

    def test_frozen_child_reuses_executable_without_external_python(self):
        with patch.object(sys, 'frozen', True, create=True):
            command = server_command(Path('/project with spaces'), 18501)
        self.assertEqual(command, [sys.executable, '--root', str(Path('/project with spaces')), '--serve', '18501'])

    def test_server_is_local_headless_and_no_file_watcher(self):
        with patch('streamlit.web.bootstrap.run') as run:
            serve(18501)
        options = run.call_args.args[3]
        self.assertEqual(options['server.address'], '127.0.0.1')
        self.assertFalse(options['global.developmentMode'])
        self.assertTrue(options['server.headless'])
        self.assertEqual(options['server.fileWatcherType'], 'none')

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

    def test_navigation_keeps_cover_until_complete_streamlit_screen(self):
        host = DesktopHost(Path('/project'))
        host.window = Mock()
        host.set_cover = Mock()
        host.stopped = Mock()
        host.stopped.wait.side_effect = [False,False]
        host.window.evaluate_js.side_effect = [False,True]
        process=Mock()
        process.poll.return_value=None
        host.wait_for_screen(process)
        self.assertEqual(host.window.evaluate_js.call_count,2)
        host.set_cover.assert_called_once_with(False)
        host.window.show.assert_called_once()

    def test_hidden_first_window_uses_matching_background_and_load_events(self):
        from src.desktop import main
        with tempfile.TemporaryDirectory() as directory, patch.object(sys,'argv',['lass','--root',directory]), \
             patch('src.windows_runtime.ensure_webview2'), patch('webview.create_window',return_value=MagicMock()) as create, \
             patch('webview.start') as start, patch('src.desktop.threading.Thread'), patch('tempfile.tempdir',tempfile.gettempdir()):
            main()
        self.assertTrue(create.call_args.kwargs['hidden'])
        self.assertEqual(create.call_args.kwargs['background_color'],'#0b0d10')
        self.assertEqual(start.call_args.kwargs['gui'],'edgechromium')
        self.assertEqual(start.call_args.kwargs['storage_path'],str(Path(directory)/'data/webview'))

    def test_temporary_files_stay_in_owned_data_directory(self):
        from src.desktop import configure_paths
        import os
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ),patch('tempfile.tempdir',tempfile.gettempdir()):
            configure_paths(Path(directory))
            with tempfile.NamedTemporaryFile() as handle:
                self.assertEqual(Path(handle.name).parent,Path(directory)/'data/tmp')
            self.assertEqual(os.environ['TEMP'],str(Path(directory)/'data/tmp'))
