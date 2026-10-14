import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
class WindowsManifestTests(unittest.TestCase):
    def test_utf8_process_codepage_preserves_korean_model_paths(self):
        root=Path(__file__).parents[1]
        document=ET.parse(root/'src/assets/lass.manifest')
        node=document.find('.//{http://schemas.microsoft.com/SMI/2019/WindowsSettings}activeCodePage')
        self.assertEqual(node.text,'UTF-8')
        self.assertIn('MinVersion=10.0.18362',(root/'scripts/installer.iss').read_text(encoding='utf-8'))
