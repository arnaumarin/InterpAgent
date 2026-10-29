"""Unit tests for the SAE input hook policy shared by FeatureExplainer and FeatureFinder.

Run:  python -m unittest InterpAgent/tests/test_sae_hook_policy.py -v
(no model download, no GPU; only the resolver and the sae-lens return-type handling)
"""
import importlib
import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPLAINER = ROOT / 'workspace/Agents/FeatureExplainer/modules'
FINDER = ROOT / 'workspace/Agents/FeatureFinder/modules/featurefinder'


class _Gemma2Layer:
    def __init__(self):
        self.mlp = object()
        self.post_feedforward_layernorm = object()


class _PlainLayer:
    def __init__(self):
        self.mlp = object()


class _FeedForwardLayer:
    def __init__(self):
        self.feed_forward = object()


def _load(module_dir, name, hook_mode=None, score_special=None):
    for key, val in (('INTERPAGENT_SAE_HOOK', hook_mode), ('INTERPAGENT_SCORE_SPECIAL_TOKENS', score_special)):
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
    sys.path.insert(0, str(module_dir))
    try:
        sys.modules.pop(name, None)
        return importlib.import_module(name)
    finally:
        sys.path.pop(0)


class HookPolicyTests(unittest.TestCase):
    def test_explainer_default_prefers_post_norm(self):
        fe = _load(EXPLAINER, 'feature_explainer')
        layer = _Gemma2Layer()
        module, name = fe.resolve_sae_hook_module(layer)
        self.assertIs(module, layer.post_feedforward_layernorm)
        self.assertEqual(name, 'post_feedforward_layernorm')
        self.assertTrue(fe.EXCLUDE_SPECIAL_TOKENS)

    def test_explainer_legacy_mode_hooks_raw_mlp(self):
        fe = _load(EXPLAINER, 'feature_explainer', hook_mode='legacy_mlp')
        layer = _Gemma2Layer()
        module, name = fe.resolve_sae_hook_module(layer)
        self.assertIs(module, layer.mlp)
        self.assertEqual(name, 'mlp')

    def test_explainer_falls_back_when_no_post_norm(self):
        fe = _load(EXPLAINER, 'feature_explainer')
        self.assertEqual(fe.resolve_sae_hook_module(_PlainLayer())[1], 'mlp')
        self.assertEqual(fe.resolve_sae_hook_module(_FeedForwardLayer())[1], 'feed_forward')
        with self.assertRaises(RuntimeError):
            fe.resolve_sae_hook_module(types.SimpleNamespace())

    def test_special_token_flag(self):
        fe = _load(EXPLAINER, 'feature_explainer', score_special='1')
        self.assertFalse(fe.EXCLUDE_SPECIAL_TOKENS)

    def test_finder_policy_matches_explainer(self):
        try:
            ff = _load(FINDER, 'completeneuralatlaspipeline')
        except ImportError as e:  # scanpy etc. not installed in this environment
            self.skipTest(f'FeatureFinder deps missing: {e}')
        layer = _Gemma2Layer()
        self.assertEqual(ff.resolve_sae_hook_module(layer)[1], 'post_feedforward_layernorm')
        self.assertEqual(ff.SAE_HOOK_MODE, 'post_feedforward_layernorm')
        self.assertTrue(ff.EXCLUDE_SPECIAL_TOKENS)

    def test_sae_lens_return_type_handling(self):
        """The initialize() code path must accept both a tuple and a bare SAE object."""
        fe = _load(EXPLAINER, 'feature_explainer')
        import inspect
        src = inspect.getsource(fe.SAEInterface.initialize)
        self.assertIn('isinstance(loaded, tuple)', src)
        self.assertNotIn('self.sae, self.sae_cfg, self.sae_sparsity = SAE.from_pretrained', src)


if __name__ == '__main__':
    unittest.main()
