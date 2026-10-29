"""FeatureFinder marker statistics: prompt-level testing, fold changes on raw SAE codes,
and a split-half validation that can fail.

Run from the repository root:  python -m unittest tests/test_featurefinder_marker_stats.py -v
"""
import importlib.util
import re
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FF = ROOT / 'workspace/Agents/FeatureFinder'
spec = importlib.util.spec_from_file_location('marker_stats', FF / 'modules/featurefinder/marker_stats.py')
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def make_prompts(X, categories):
    obs = pd.DataFrame({'prompt_id': [f'p{i}' for i in range(len(categories))], 'category': categories})
    adata = ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=obs,
                       var=pd.DataFrame(index=[f'F{j}' for j in range(X.shape[1])]))
    return ms.prompt_level(adata)


def planted(n_per_cat=40, n_features=50, seed=0):
    """Two categories; F0 is a real marker of 'a' (prompt-level shift), the rest are noise."""
    rng = np.random.default_rng(seed)
    categories = ['a'] * n_per_cat + ['b'] * n_per_cat
    X = rng.exponential(1.0, size=(2 * n_per_cat, n_features))
    X[:n_per_cat, 0] += 3.0
    return make_prompts(X, categories)


class LogFoldChangeTests(unittest.TestCase):
    def test_logfc_is_log2_ratio_of_raw_means(self):
        # group mean 3 vs rest mean 1: true log2FC = 1.585 (scanpy's expm1 back-transform gives 3.47)
        X = np.array([[3.0], [3.0], [3.0], [1.0], [1.0], [1.0]])
        X = np.hstack([X, np.ones((6, 1))])
        pdata = ms.rank_markers(make_prompts(X, ['a'] * 3 + ['b'] * 3), n_genes=2)
        df = ms.select_markers(pdata, 'a', {'max_pval_adj': 1.0, 'min_score': -np.inf,
                                            'min_logfc': -np.inf, 'min_effect_size': -np.inf})
        row = df.set_index('names').loc['F0']
        self.assertAlmostEqual(row['logfoldchanges'], np.log2(3.0), places=4)
        self.assertAlmostEqual(row['mean_in_category'], 3.0, places=5)
        uns_lfc = pdata.uns['rank_genes_groups']['logfoldchanges']['a']
        uns_names = list(pdata.uns['rank_genes_groups']['names']['a'])
        self.assertAlmostEqual(float(uns_lfc[uns_names.index('F0')]), np.log2(3.0), places=4)

    def test_small_shift_does_not_pass_logfc_filter(self):
        # 3.2 vs 2.8: log2FC = 0.19, below the 0.5 threshold (scanpy reported 0.61)
        rng = np.random.default_rng(0)
        X = np.concatenate([3.2 + 0.05 * rng.standard_normal(30), 2.8 + 0.05 * rng.standard_normal(30)])[:, None]
        pdata = ms.rank_markers(make_prompts(X, ['a'] * 30 + ['b'] * 30), n_genes=1)
        self.assertEqual(len(ms.select_markers(pdata, 'a')), 0)


class PromptLevelTests(unittest.TestCase):
    def test_tokens_are_averaged_per_prompt(self):
        obs = pd.DataFrame({'prompt_id': ['x', 'x', 'y', 'y', 'y'], 'category': ['a', 'a', 'b', 'b', 'b']})
        X = np.array([[1.0], [3.0], [0.0], [3.0], [6.0]], dtype=np.float32)
        adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=['F0']))
        adata.raw = adata
        adata.X = np.log1p(adata.X)  # normalized matrix must not be used
        pdata = ms.prompt_level(adata)
        self.assertEqual(pdata.n_obs, 2)
        np.testing.assert_allclose(pdata.X[:, 0], [2.0, 3.0])
        self.assertEqual(list(pdata.obs['category'].astype(str)), ['a', 'b'])
        self.assertEqual(list(pdata.obs['n_tokens']), [2, 3])

    def test_prompt_spanning_categories_is_rejected(self):
        obs = pd.DataFrame({'prompt_id': ['x', 'x'], 'category': ['a', 'b']})
        adata = ad.AnnData(X=np.ones((2, 1), dtype=np.float32), obs=obs, var=pd.DataFrame(index=['F0']))
        with self.assertRaises(ValueError):
            ms.prompt_level(adata)

    def test_many_tokens_per_prompt_do_not_create_markers(self):
        # F0 varies between prompts but not between categories; 60 tokens per prompt would make
        # a token-level Wilcoxon highly significant whenever the prompt offsets differ by chance.
        rng = np.random.default_rng(1)
        n_prompts, n_tokens = 30, 60
        offsets = rng.exponential(1.0, size=2 * n_prompts)
        rows, prompt_ids, cats = [], [], []
        for p, off in enumerate(offsets):
            rows.append(off + 0.01 * rng.random((n_tokens, 2)))
            prompt_ids += [f'p{p}'] * n_tokens
            cats += ['a' if p < n_prompts else 'b'] * n_tokens
        adata = ad.AnnData(X=np.vstack(rows).astype(np.float32),
                           obs=pd.DataFrame({'prompt_id': prompt_ids, 'category': cats}),
                           var=pd.DataFrame(index=['F0', 'F1']))
        pdata = ms.rank_markers(ms.prompt_level(adata), n_genes=2)
        self.assertEqual(pdata.n_obs, 2 * n_prompts)
        self.assertEqual(len(ms.select_markers(pdata, 'a')), 0)
        self.assertEqual(len(ms.select_markers(pdata, 'b')), 0)


class SplitHalfValidationTests(unittest.TestCase):
    def test_planted_marker_replicates(self):
        res = ms.split_half_validation(planted())
        self.assertTrue(res['a']['valid'])
        self.assertIn('F0', res['a']['replicated_markers'])
        self.assertEqual(res['a']['n_discovery_prompts'] + res['a']['n_heldout_prompts'], 40)

    def test_category_without_markers_fails(self):
        res = ms.split_half_validation(planted())
        self.assertFalse(res['b']['valid'])

    def test_chance_markers_do_not_replicate(self):
        # Pure noise with lenient selection: the discovery half finds chance "markers",
        # which must fail on the held-out half.
        rng = np.random.default_rng(2)
        pdata = make_prompts(rng.exponential(1.0, size=(80, 400)), ['a'] * 40 + ['b'] * 40)
        lenient = {'max_pval_adj': 1.0, 'min_score': 0.0, 'min_logfc': 0.0, 'min_effect_size': 0.3}
        res = ms.split_half_validation(pdata, thresholds=lenient, n_genes=400)
        for category in ('a', 'b'):
            self.assertGreater(res[category]['n_discovery_markers'], 0)
            self.assertFalse(res[category]['valid'])

    def test_too_few_prompts_fails(self):
        res = ms.split_half_validation(make_prompts(np.ones((4, 3)), ['a', 'a', 'b', 'b']))
        self.assertFalse(res['a']['valid'])
        self.assertIn('Too few prompts', res['a']['reason'])


class PermutationTests(unittest.TestCase):
    def test_permutes_prompt_labels(self):
        res = ms.permutation_test(planted(n_features=20), n_permutations=9, n_genes=20)
        self.assertEqual(res['unit'], 'prompt')
        self.assertLessEqual(res['permutation_pvalues']['a'], 0.1 + 1e-9)


class DefaultsTests(unittest.TestCase):
    def test_per_prompt_statistic_is_default(self):
        pipeline = (FF / 'modules/featurefinder/completeneuralatlaspipeline.py').read_text()
        self.assertIn('token_statistic="mean",', pipeline)
        self.assertIn('token_statistic: str = "mean",', (FF / 'tools.py').read_text())
        self.assertNotRegex(pipeline, re.compile(r"rank_genes_groups\([^)]*use_raw=True", re.S))


if __name__ == '__main__':
    unittest.main()
