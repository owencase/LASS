import unittest
from unittest.mock import patch
from src.windows_runtime import ensure_webview2, BOOTSTRAPPER_URL


class WindowsRuntimeTests(unittest.TestCase):
    def test_existing_webview_runtime_needs_no_installer(self):
        with patch('src.windows_runtime.webview2_installed', return_value=True), \
             patch('src.windows_runtime.urlretrieve') as download:
            ensure_webview2()
            download.assert_not_called()

    def test_missing_webview_runtime_uses_official_silent_installer(self):
        with patch('src.windows_runtime.webview2_installed', side_effect=[False, True]), \
             patch('src.windows_runtime.urlretrieve') as download, \
             patch('src.windows_runtime.subprocess.run') as run:
            ensure_webview2()
        self.assertEqual(download.call_args.args[0], BOOTSTRAPPER_URL)
        self.assertEqual(run.call_args.args[0][1:], ['/silent', '/install'])
        self.assertTrue(run.call_args.kwargs['check'])
