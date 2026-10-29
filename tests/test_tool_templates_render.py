"""Every FeatureExplainer tool must render its generated-code template and the result
must be valid Python. Guards against unescaped braces / raw escape sequences inside the
f-string templates (which previously made `explain_feature` and
`load_feature_data_from_results` un-renderable).

Run from the repository root:  python -m unittest InterpAgent/tests/test_tool_templates_render.py -v
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'workspace/Agents/FeatureExplainer/tools.py'
DUMMY = {'feature_idx': 1234, 'results_dir': '/tmp/results_saeL0', 'max_iterations': 5}


def _load_tools():
    spec = importlib.util.spec_from_file_location('fe_tools_under_test', str(TOOLS))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ToolTemplateRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_tools()
        cls.tools = [(n, t) for n, t in vars(cls.module).items() if hasattr(t, 'invoke') and hasattr(t, 'args_schema')]

    def test_all_tools_render_and_compile(self):
        self.assertGreaterEqual(len(self.tools), 5)
        for name, tool in self.tools:
            with self.subTest(tool=name):
                fields = list(tool.args_schema.model_fields.keys())
                args = {k: DUMMY[k] for k in fields if k in DUMMY}
                code = tool.invoke(args)
                self.assertIsInstance(code, str)
                compile(code, f'<{name}>', 'exec')

    def test_explain_feature_uses_calibrated_threshold_and_requested_iterations(self):
        code = self.module.explain_feature.invoke({'feature_idx': 7, 'max_iterations': 5})
        self.assertIn('max_iterations = 5', code)
        self.assertIn('calibrated_sae_threshold(', code)
        self.assertIn('SAEInterface(layer=sae_layer, feature_idx=feature_idx, threshold=sae_threshold)', code)
        # runtime placeholders must survive as single-brace f-string fields in the generated code
        self.assertIn('{sae_layer}', code)
        self.assertNotIn('{{', code)

    def test_test_case_total_is_used_and_explicit_per_class_override_is_recorded(self):
        for arguments, expected in [({'feature_idx': 7}, (24, 6)),
                                    ({'feature_idx': 7, 'n_test_cases': 12}, (12, 3)),
                                    ({'feature_idx': 7, 'cases_per_type': 5}, (20, 5))]:
            code = self.module.explain_feature.invoke(arguments)
            config = code[code.index('feature_idx = '):code.index('# Configure logging verbosity')]
            state = {}
            exec(config, state)
            self.assertEqual((state['n_test_cases'], state['cases_per_type']), expected)


if __name__ == '__main__':
    unittest.main()
