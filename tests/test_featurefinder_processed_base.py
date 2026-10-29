"""GEMMA_SAE_PROCESSED_BASE must redirect the FeatureFinder processed-data directory without changing the default."""
import os, re, unittest
from pathlib import Path
SRC = Path(__file__).resolve().parents[1] / 'workspace/Agents/FeatureFinder/modules/featurefinder/completeneuralatlaspipeline.py'

class ProcessedBaseTests(unittest.TestCase):
    def _expr(self):
        src = SRC.read_text()
        m = re.search(r'processed_base = os\.environ\.get\("GEMMA_SAE_PROCESSED_BASE", "([^"]+)"\)', src)
        self.assertIsNotNone(m); return m.group(1)
    def test_default_is_historical_location(self):
        self.assertEqual(self._expr(), '/mnt/Storage7/atlas4llm/gemma2SAE')
    def test_override_changes_path(self):
        src = SRC.read_text()
        self.assertIn('self.processed_data_dir = f"{processed_base}/results_multitoken/', src)
        os.environ['GEMMA_SAE_PROCESSED_BASE'] = '/tmp/x/'
        base = os.environ.get("GEMMA_SAE_PROCESSED_BASE", "/mnt/Storage7/atlas4llm/gemma2SAE").rstrip("/")
        self.assertEqual(base, '/tmp/x'); del os.environ['GEMMA_SAE_PROCESSED_BASE']
if __name__ == '__main__': unittest.main()
