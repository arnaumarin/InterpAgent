#!/usr/bin/env python3
"""
Complete Neural Atlas Pipeline for Gemma 2B + SAE encoding (single layer, no fallback)

- Choose exactly one SAE layer with --sae-layer
- Hook the chosen layer's MLP output and encode with SAE (sae.encode(x))
- NEVER replaces SAE codes with raw MLP activations; if SAE is missing or encode fails -> STOP
- Produces markers/visualizations for the chosen layer's SAE features

Usage:
  python complete_neural_atlas_pipeline_multitoken_sae_single_layer.py --sae-layer 0 [--sae-width width_16k] [--sae-variant canonical] [--sae-release gemma-scope-2b-pt-mlp-canonical] [model_key]

To run for all layers (example: 26 layers) in bash:
  for L in $(seq 0 25); do
    python complete_neural_atlas_pipeline_multitoken_sae_single_layer.py --sae-layer $L
  done
"""

import os
import sys
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import scanpy as sc
import anndata as ad
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from sae_lens import SAE  # <- pre-installed

try:
    from . import marker_stats
except ImportError:  # run as a script
    import marker_stats

# Token statistics. "all_tokens" keeps one row per token (for embeddings/plots); marker
# statistics are always computed on one row per prompt (tokens are averaged per prompt).
TOKEN_STATISTICS = ("mean", "max", "last_token", "all_tokens", "mean_last_k", "max_last_k", "top_p_trimmed_mean")


# --- SAE input hook policy ----------------------------------------------------
# GemmaScope MLP dictionaries are trained on TransformerLens `blocks.L.hook_mlp_out`,
# which on Gemma-2 is the MLP output *after* `post_feedforward_layernorm`. Hooking the
# raw HF `mlp` module output (the historical behaviour) feeds the pre-norm vector to the
# encoder. The default now hooks the post-norm module whenever the architecture has one.
# Set INTERPAGENT_SAE_HOOK=legacy_mlp only to reproduce historical runs. Special tokens
# (BOS/EOS/pad) are excluded from the activation matrix by default: the BOS activation is
# text-independent and carries no category signal.
SAE_HOOK_MODE = os.environ.get("INTERPAGENT_SAE_HOOK", "post_feedforward_layernorm")
EXCLUDE_SPECIAL_TOKENS = os.environ.get("INTERPAGENT_SCORE_SPECIAL_TOKENS", "0") != "1"


def resolve_sae_hook_module(layer_module):
    """Return (module, name) whose forward output is the SAE input for this layer."""
    if SAE_HOOK_MODE != "legacy_mlp" and hasattr(layer_module, "post_feedforward_layernorm"):
        return layer_module.post_feedforward_layernorm, "post_feedforward_layernorm"
    for name in ("mlp", "feed_forward", "block_sparse_moe"):
        if hasattr(layer_module, name):
            return getattr(layer_module, name), name
    raise RuntimeError("No MLP-like submodule found to hook for SAE input")

# Configure scanpy
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=300, facecolor='white')


class CompleteNeuralAtlasPipeline:
    """
    Pipeline for neural atlas analysis using SAE-encoded features from ONE chosen layer.
    """

    def __init__(
        self,
        model_key="gemma_2b",
        max_prompts_per_category=1000,
        token_statistic="mean",
        last_k_tokens=10,
        top_p_trim=0.5,
        run_permutation_test=False,
        n_permutations=50,
        sae_release="gemma-scope-2b-pt-mlp-canonical",
        sae_width_dir="width_16k",
        sae_variant="canonical",
        sae_layer_idx=None,
        concepts=None,
        prompt_files=None):
        # ----- Model configuration -----
        import sys
        import os
        # Add agent directory to path to import model_config
        agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        from model_config import get_model_config, get_model_revision, get_output_dir_multitokenwEnglish

        if sae_layer_idx is None:
            raise RuntimeError("You must provide a valid --sae-layer index (integer).")

        self.model_key = model_key
        self.model_config = get_model_config(model_key)
        self.model_name = self.model_config["name"]
        self.training_stage = self.model_config["training_stage"]
        self.model_revision = get_model_revision(model_key)

        if token_statistic not in TOKEN_STATISTICS:
            raise ValueError(f"Unknown token_statistic {token_statistic!r}; expected one of {TOKEN_STATISTICS}")
        self.max_prompts_per_category = max_prompts_per_category
        self.token_statistic = token_statistic
        self.last_k_tokens = last_k_tokens
        self.top_p_trim = top_p_trim
        self.run_permutation_test = run_permutation_test
        self.n_permutations = n_permutations

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----- SAE configuration -----
        self.sae_release = sae_release
        self.sae_width_dir = sae_width_dir
        self.sae_variant = sae_variant
        self.sae_layer_idx = int(sae_layer_idx)  # one layer per run
        self.sae = None
        self.sae_cfg = None
        self.sae_sparsity = None
        self.d_sae = None  # feature count

        # ----- Output directory -----
        base_output_dir = get_output_dir_multitokenwEnglish(model_key)
        # Historical directory names are kept for the original statistics.
        if self.token_statistic == "all_tokens":
            token_suffix = "666_strict"
        elif self.token_statistic in ("mean_last_k", "max_last_k", "top_p_trimmed_mean"):
            token_suffix = str(self.last_k_tokens)
        else:
            token_suffix = self.token_statistic
        
        # Separate directories for different data types
        self.output_dir = f"{base_output_dir}_{token_suffix}_saeL{self.sae_layer_idx}_{self.sae_width_dir}"  # For figures and CSVs
        # Large processed data: base directory is configurable (GEMMA_SAE_PROCESSED_BASE); the
        # default keeps the historical machine-specific location so existing runs are unchanged.
        processed_base = os.environ.get("GEMMA_SAE_PROCESSED_BASE", "/mnt/Storage7/atlas4llm/gemma2SAE").rstrip("/")
        self.processed_data_dir = f"{processed_base}/results_multitoken/google_gemma_2_2b_{token_suffix}_saeL{self.sae_layer_idx}_{self.sae_width_dir}/processed_data"
        
        # Create both directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # Define consistent category ordering for all visualizations
        # Store the prompt file specification
        self.prompt_files = prompt_files  # Dict or None
        self.concepts = concepts  # List or None
        
        # All available standard concepts
        all_available_concepts = [
            # Germanic
            'english', 'german',
            # Romance
            'french', 'italian', 'spanish', 'portuguese',
            # Asian
            'chinese', 'japanese', 'indonesian'
        ]
        
        # Determine category order based on what was specified
        if prompt_files is not None:
            # Use custom prompt files - extract category names from dict keys
            self.consistent_category_order = list(prompt_files.keys())
            print(f"Using custom prompt files: {len(self.consistent_category_order)} categories")
        elif concepts is not None:
            # Use specified concepts with standard file naming
            invalid_concepts = [c for c in concepts if c not in all_available_concepts]
            if invalid_concepts:
                raise ValueError(
                    f"Invalid concept(s): {invalid_concepts}. "
                    f"Available concepts: {all_available_concepts}"
                )
            self.consistent_category_order = concepts
        else:
            # Default to all standard concepts
            self.consistent_category_order = all_available_concepts

        print("=" * 80)
        print(f"COMPLETE NEURAL ATLAS PIPELINE (SAE, ONE LAYER) - {self.training_stage.upper()}")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        if self.model_revision:
            print(f"Checkpoint: {self.model_revision}")
        print(f"Device: {self.device}")
        print(f"SAE release: {self.sae_release}")
        print(f"SAE layer: {self.sae_layer_idx}")
        print(f"SAE width dir: {self.sae_width_dir}")
        print(f"SAE variant: {self.sae_variant}")
        print(f"Concepts: {', '.join(self.consistent_category_order)} ({len(self.consistent_category_order)} total)")
        print(f"Max prompts/category: {max_prompts_per_category}")
        print(f"Token statistic: {token_statistic}")
        if token_statistic in ["mean_last_k", "max_last_k", "top_p_trimmed_mean"]:
            print(f"Last k tokens: {last_k_tokens}")
        if token_statistic == "top_p_trimmed_mean":
            print(f"Top-p trim: {top_p_trim}")
        print(f"Output directory (figures/CSVs): {self.output_dir}")
        print(f"Processed data directory: {self.processed_data_dir}")

        # ----- Load model -----
        from transformers import AutoConfig
        if self.model_revision:
            print(f"Loading {self.model_name} checkpoint {self.model_revision}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.model_revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
            )
        else:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
            )

        if all(p.device.type == "cpu" for p in self.model.parameters()):
            self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Architecture info
        self.n_layers = getattr(self.model.config, 'num_hidden_layers', None) or getattr(self.model.config, 'n_layer', 26)
        self.d_model = getattr(self.model.config, 'hidden_size', None) or getattr(self.model.config, 'n_embd', 2304)
        print(f"Architecture: {self.n_layers} layers × d_model={self.d_model}")
        if not (0 <= self.sae_layer_idx < self.n_layers):
            raise RuntimeError(f"--sae-layer {self.sae_layer_idx} is out of range [0, {self.n_layers-1}]")

        # ----- Load exactly one SAE (no fallback allowed) -----
        self._load_single_sae_or_die()

        # Results storage
        self.results = {
            'metadata': {
                'model_key': model_key,
                'model_name': self.model_name,
                'training_stage': self.training_stage,
                'device': str(self.device),
                'sae_release': self.sae_release,
                'sae_width_dir': self.sae_width_dir,
                'sae_variant': self.sae_variant,
                'sae_layer': self.sae_layer_idx,
                'd_sae': int(self.d_sae),
                'sparsity_estimate': float(self.sae_sparsity) if self.sae_sparsity is not None else None,
                'max_prompts_per_category': max_prompts_per_category,
                'token_statistic': token_statistic,
                'last_k_tokens': last_k_tokens,
                'top_p_trim': top_p_trim,
            }
        }

        print(f"✅ {self.model_name} + SAE(layer={self.sae_layer_idx}) initialized successfully")

    # ---------- SAE helpers ----------
    def _sae_id_for_layer(self, layer_idx: int) -> str:
        return f"layer_{layer_idx}/{self.sae_width_dir}/{self.sae_variant}"

    def _infer_d_sae(self, sae, cfg_dict=None):
        # best effort: prefer attribute; else W_dec shape max-dim; else defer until first encode
        try:
            if hasattr(sae, 'd_sae') and isinstance(sae.d_sae, int):
                return sae.d_sae
        except Exception:
            pass
        try:
            if cfg_dict and 'd_sae' in cfg_dict:
                return int(cfg_dict['d_sae'])
        except Exception:
            pass
        try:
            if hasattr(sae, 'W_dec'):
                shape = tuple(sae.W_dec.shape)
                return max(shape)
        except Exception:
            pass
        return None

    def _load_single_sae_or_die(self):
        sae_id = self._sae_id_for_layer(self.sae_layer_idx)
        print("\n[SAE] Loading SAE for the chosen layer only...")
        print(f"  -> release='{self.sae_release}', sae_id='{sae_id}'")
        try:
            loaded = SAE.from_pretrained(
                release=self.sae_release,
                sae_id=sae_id,
            )
            # sae-lens < 5 returned (sae, cfg_dict, sparsity); newer versions return the SAE only.
            if isinstance(loaded, tuple):
                sae, cfg_dict, sparsity = loaded
            else:
                sae, cfg_dict, sparsity = loaded, getattr(loaded, "cfg", None), None
            # Keep SAE on CPU for now; we move it to the correct device inside the hook per-batch
            sae.eval()
            self.sae = sae
            self.sae_cfg = cfg_dict
            self.sae_sparsity = sparsity
            self.d_sae = self._infer_d_sae(sae, cfg_dict)
            sparsity_str = f"{float(sparsity):.4f}" if sparsity is not None else "unknown"
            print(f"[SAE] Loaded OK | Estimated d_sae={self.d_sae} | sparsity≈{sparsity_str}")
        except Exception as e:
            raise RuntimeError(f"[SAE][FATAL] Could not load SAE '{sae_id}': {e}")

    # ---------- Aggregation helpers ----------
    def compute_token_statistic(self, token_activations, attention_mask, token_pos):
        full_activation = torch.cat(token_activations, dim=0).numpy()
        return full_activation

    def aggregate_sequence_activations(self, sequence_activations, attention_mask):
        if not sequence_activations:
            return None
        seq_array = np.array(sequence_activations)  # [valid_seq, d_sae]
        valid_mask = attention_mask.detach().cpu().numpy().flatten()
        valid_activations = seq_array[valid_mask == 1]
        if len(valid_activations) == 0:
            return np.zeros(seq_array.shape[1])
        if self.token_statistic == "mean":
            return np.mean(valid_activations, axis=0)
        elif self.token_statistic == "max":
            return np.max(valid_activations, axis=0)
        elif self.token_statistic == "last_token":
            return valid_activations[-1]
        elif self.token_statistic == "mean_last_k":
            k = min(self.last_k_tokens, len(valid_activations))
            return np.mean(valid_activations[-k:], axis=0)
        elif self.token_statistic == "max_last_k":
            k = min(self.last_k_tokens, len(valid_activations))
            return np.max(valid_activations[-k:], axis=0)
        elif self.token_statistic == "top_p_trimmed_mean":
            k = min(self.last_k_tokens, len(valid_activations))
            last_k_activations = valid_activations[-k:]
            trimmed_means = []
            for neuron_idx in range(last_k_activations.shape[1]):
                neuron_acts = last_k_activations[:, neuron_idx]
                n_top = max(1, int(len(neuron_acts) * self.top_p_trim))
                top_vals = np.partition(neuron_acts, -n_top)[-n_top:]
                trimmed_means.append(np.mean(top_vals))
            return np.array(trimmed_means)
        else:
            raise ValueError(f"Unknown token_statistic {self.token_statistic!r}")

    def load_prompts_and_extract_activations(self, prompts_dir, processed_data_dir):
        """
        Load prompts and extract SAE codes for the chosen layer.
        """
        print(f"\nSTEP 1: LOADING PROMPTS AND EXTRACTING SAE CODES (layer={self.sae_layer_idx})")
        print("-" * 75)

        # ----- Load prompts -----
        categorized_prompts = {}
        
        # Determine which files to load
        if self.prompt_files is not None:
            # Use custom prompt files (already full paths)
            category_files = self.prompt_files
        else:
            # Use standard naming convention with prompts_dir
            category_files = {
                # Germanic
                'english': os.path.join(prompts_dir, 'prompts_english.txt'),
                'german': os.path.join(prompts_dir, 'prompts_german.txt'),
                # Romance
                'french': os.path.join(prompts_dir, 'prompts_french.txt'),
                'italian': os.path.join(prompts_dir, 'prompts_italian.txt'),
                'spanish': os.path.join(prompts_dir, 'prompts_spanish.txt'),
                'portuguese': os.path.join(prompts_dir, 'prompts_portuguese.txt'),
                # Asian
                'chinese': os.path.join(prompts_dir, 'prompts_chinese.txt'),
                'japanese': os.path.join(prompts_dir, 'prompts_japanese.txt'),
                'indonesian': os.path.join(prompts_dir, 'prompts_indonesian.txt')
            }

        for category in self.consistent_category_order:
            if category not in category_files:
                print(f"  Skipping {category}: not in category_files")
                continue
            filepath = category_files[category]
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    all_lines = [line.strip() for line in f if line.strip()]
                total_available = len(all_lines)
                if total_available > self.max_prompts_per_category:
                    prompts = all_lines[:self.max_prompts_per_category]
                    print(f"  {category.title()}: {len(prompts)} prompts (limited from {total_available})")
                else:
                    prompts = all_lines
                    print(f"  {category.title()}: {len(prompts)} prompts (all available)")
                categorized_prompts[category] = prompts

        # ----- Storage -----
        all_activations = []
        all_prompts = []
        all_categories = []
        all_token_positions = []
        all_token_texts = []
        all_seq_lens = []
        all_token_offsets = []
        all_prompt_ids = []  # one id per prompt (not per prompt text), shared by its tokens
        prompt_counter = 0

        total_prompts = sum(len(p) for p in categorized_prompts.values())
        print(f"\nExtracting SAE codes (layer {self.sae_layer_idx}) for {total_prompts} prompts...")

        codes_captured_once = False  # for shape debug

        # Prepare hook for the ONE chosen layer
        target_layer_idx = self.sae_layer_idx

        for category in self.consistent_category_order:
            if category not in categorized_prompts:
                continue

            prompts = categorized_prompts[category]
            print(f"Processing {category} prompts...")

            for i, prompt in enumerate(tqdm(prompts, desc=f"Extracting {category}")):
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_attention_mask=True,
                    return_special_tokens_mask=True
                )
                special_tokens_mask = inputs.pop("special_tokens_mask")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Captured codes for this prompt
                captured_codes = {"codes": None}

                def hook_fn(module, input, output):
                    # Some HF modules return tuple; normalize
                    x = output[0] if isinstance(output, (tuple, list)) else output  # [b, seq, d_model]
                    # Move SAE to same device as x (we do this once per forward if needed)
                    if self.sae.device != x.device:
                        self.sae.to(x.device)
                    try:
                        with torch.no_grad():
                            x32 = x.to(dtype=torch.float32) if x.dtype != torch.float32 else x
                            codes = self.sae.encode(x32)  # [b, seq, d_sae]
                        captured_codes["codes"] = codes.detach().cpu()
                    except Exception as e:
                        raise RuntimeError(f"[SAE][ENCODE-FAIL] layer={target_layer_idx}: {e}")

                # Locate target module to hook
                layer_module = self.model.model.layers[target_layer_idx]
                target_module, target_name = resolve_sae_hook_module(layer_module)
                if not codes_captured_once:
                    print(f"[HOOK] SAE input = model.layers[{target_layer_idx}].{target_name} (mode={SAE_HOOK_MODE}; special tokens {'excluded' if EXCLUDE_SPECIAL_TOKENS else 'included'})")

                # Register one hook
                hook = target_module.register_forward_hook(hook_fn)

                # Forward
                with torch.no_grad():
                    _ = self.model(**inputs)

                # Remove hook
                hook.remove()

                # Validate capture
                if captured_codes["codes"] is None:
                    raise RuntimeError(f"[SAE][FATAL] No codes captured for layer {target_layer_idx}; hook target={target_name}")

                codes_cpu = captured_codes["codes"]  # [1, seq, d_sae]
                seq_len = int(codes_cpu.shape[1])
                d_out = int(codes_cpu.shape[-1])
                if not codes_captured_once:
                    print(f"[DEBUG] First capture: codes shape = {tuple(codes_cpu.shape)} (b, seq, d_sae={d_out}) from layer={target_layer_idx}.{target_name}")
                    if self.d_sae is None:
                        self.d_sae = d_out
                    elif self.d_sae != d_out:
                        raise RuntimeError(f"[SAE][FATAL] Inconsistent d_sae: expected {self.d_sae}, got {d_out}")
                    codes_captured_once = True

                attention_mask = inputs['attention_mask']
                valid_tokens = int(attention_mask.sum().item())

                # Token metadata
                token_ids = inputs['input_ids'][0].detach().cpu().numpy()
                token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
                prompt_id = f"{category}_{prompt_counter}"
                prompt_counter += 1

                # Collect features
                if self.token_statistic == "all_tokens":
                    for token_pos in range(seq_len):
                        if attention_mask[0, token_pos] == 0:
                            continue
                        if EXCLUDE_SPECIAL_TOKENS and int(special_tokens_mask[0, token_pos]) == 1:
                            continue
                        token_feat = torch.as_tensor(codes_cpu[0, token_pos, :]).float().numpy()  # [d_sae]
                        all_activations.append(token_feat)
                        all_prompts.append(prompt[:100] + "..." if len(prompt) > 100 else prompt)
                        all_categories.append(category)
                        all_token_positions.append(token_pos)
                        all_token_texts.append(token_texts[token_pos])
                        all_seq_lens.append(valid_tokens)
                        all_token_offsets.append(token_pos)
                        all_prompt_ids.append(prompt_id)
                else:
                    sequence_activations = []
                    for token_pos in range(seq_len):
                        if attention_mask[0, token_pos] == 0:
                            continue
                        if EXCLUDE_SPECIAL_TOKENS and int(special_tokens_mask[0, token_pos]) == 1:
                            continue
                        token_feat = torch.as_tensor(codes_cpu[0, token_pos, :]).float().numpy()
                        sequence_activations.append(token_feat)
                    if sequence_activations:
                        kept_mask = torch.ones(1, len(sequence_activations), dtype=attention_mask.dtype)
                        aggregated = self.aggregate_sequence_activations(sequence_activations, kept_mask)
                        all_activations.append(aggregated)
                        all_prompts.append(prompt[:100] + "..." if len(prompt) > 100 else prompt)
                        all_categories.append(category)
                        all_token_positions.append(-1)
                        all_token_texts.append("[AGGREGATED]")
                        all_seq_lens.append(valid_tokens)
                        all_token_offsets.append(-1)
                        all_prompt_ids.append(prompt_id)

        # To matrix
        activation_matrix = np.array(all_activations)
        prompt_ids = all_prompt_ids
        n_unique_prompts = len(set(prompt_ids))

        print(f"\nActivation matrix (SAE codes, layer={self.sae_layer_idx}) extracted:")
        print(f"  Shape: {activation_matrix.shape}")
        if self.token_statistic == "all_tokens":
            print(f"  {activation_matrix.shape[0]} tokens × {activation_matrix.shape[1]} SAE features")
            print(f"  Unique prompts: {n_unique_prompts}")
            if n_unique_prompts > 0:
                print(f"  Avg tokens per prompt: {activation_matrix.shape[0] / n_unique_prompts:.1f}")
        else:
            print(f"  {activation_matrix.shape[0]} observations × {activation_matrix.shape[1]} SAE features")
            print(f"  Statistic: {self.token_statistic}")

        prompt_metadata = pd.DataFrame({
            'prompt': all_prompts,
            'category': all_categories,
            'token_position': all_token_positions,
            'token_text': all_token_texts,
            'seq_len': all_seq_lens,
            'token_offset': all_token_offsets,
            'prompt_id': prompt_ids,
            'token_id': [f"{pid}_t{pos}" for pid, pos in zip(prompt_ids, all_token_positions)]
        })

        # Feature metadata
        if self.d_sae is None:
            raise RuntimeError("[SAE][FATAL] d_sae is unknown after encoding; cannot build feature metadata.")
        feature_names = [f"L{self.sae_layer_idx}_F{i}" for i in range(int(self.d_sae))]
        neuron_metadata = pd.DataFrame({
            'neuron_id': feature_names,
            'layer': [self.sae_layer_idx] * int(self.d_sae),
            'position': list(range(int(self.d_sae))),
        })

        # Dimension check
        if activation_matrix.shape[1] != len(feature_names):
            raise RuntimeError(
                f"[FATAL] Feature dimension mismatch: matrix has {activation_matrix.shape[1]}, "
                f"metadata lists {len(feature_names)}."
            )

        # Save intermediate
        print(f"\nSaving intermediate results to {processed_data_dir}/ ...")
        os.makedirs(processed_data_dir, exist_ok=True)
        np.save(os.path.join(processed_data_dir, 'activation_matrix.npy'), activation_matrix)
        prompt_metadata.to_csv(os.path.join(processed_data_dir, 'prompt_metadata.csv'), index=False)
        neuron_metadata.to_csv(os.path.join(processed_data_dir, 'neuron_metadata.csv'), index=False)
        with open(os.path.join(processed_data_dir, 'categorized_prompts.json'), 'w') as f:
            json.dump(categorized_prompts, f, indent=2)

        self.results['step1_data_extraction'] = {
            'activation_matrix_shape': activation_matrix.shape,
            'n_prompts_by_category': {cat: len(prompts) for cat, prompts in categorized_prompts.items()},
            'total_observations': len(all_prompts),
            'total_features': activation_matrix.shape[1],
            'sae_layer': self.sae_layer_idx,
            'd_sae': int(self.d_sae)
        }

        print("✅ Step 1 complete - SAE-coded data saved")
        return activation_matrix, prompt_metadata, neuron_metadata

    # ---------- Scanpy & downstream (logic same; features == SAE codes) ----------
    def run_scanpy_analysis(self, activation_matrix, prompt_metadata, neuron_metadata, processed_data_dir, algorithm: str = "leiden"):
        # `algorithm` selects the community-detection step (pluggable):
        #   'leiden'  -> sc.tl.leiden (default; igraph/leidenalg backend)
        #   'louvain' -> sc.tl.louvain(flavor='taynaud')  (python-louvain backend)
        # Result is mirrored to adata.obs['leiden'] for backward-compat with downstream code.
        print(f"\nSTEP 2: SCANPY DIFFERENTIAL EXPRESSION ANALYSIS (SAE features, layer={self.sae_layer_idx})")
        print("-" * 60)

        adata = ad.AnnData(X=activation_matrix, obs=prompt_metadata, var=neuron_metadata)
        adata.var.index = neuron_metadata['neuron_id']

        print(f"Created AnnData object:")
        print(f"  {adata.n_obs} observations")
        print(f"  {adata.n_vars} SAE features (layer {self.sae_layer_idx})")
        print(f"  Categories: {adata.obs['category'].unique()}")

        adata.raw = adata

        print("\n2.1 Data preprocessing...")
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=1.0, neginf=0.0)
        sc.pp.normalize_total(adata, target_sum=1e4)
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=1.0, neginf=0.0)
        sc.pp.log1p(adata)
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=1.0, neginf=0.0)

        print("2.2 Finding highly variable features...")
        try:
            sc.pp.highly_variable_genes(adata, min_mean=0.001, max_mean=10, min_disp=0.1)
            if 'highly_variable' not in adata.var.columns or adata.var['highly_variable'].fillna(False).sum() < 10:
                print("   ⚠️  Too few HV features, using all")
                adata.var['highly_variable'] = True
        except Exception as e:
            print(f"   ⚠️  HVG failed: {e}, using all features")
            adata.var['highly_variable'] = True

        n_hvg = int(adata.var['highly_variable'].sum())
        print(f"   Selected {n_hvg} highly variable features")

        print("2.3 Principal component analysis...")
        n_samples = adata.n_obs
        n_components = max(1, min(200, n_samples - 1))
        try:
            print("   Trying ARPACK solver...")
            sc.tl.pca(adata, svd_solver='arpack', n_comps=n_components, use_highly_variable=True)
            print(f"   Computed {n_components} PCs with ARPACK")
        except Exception as e:
            print(f"   ARPACK failed: {e}")
            try:
                print("   Trying LAPACK solver...")
                sc.tl.pca(adata, svd_solver='full', n_comps=n_components, use_highly_variable=True)
                print(f"   Computed {n_components} PCs with LAPACK")
            except Exception as e2:
                print(f"   LAPACK failed: {e2}")
                print("   Trying randomized solver...")
                sc.tl.pca(adata, svd_solver='randomized', n_comps=n_components, use_highly_variable=True)
                print(f"   Computed {n_components} PCs with randomized solver")

        print("2.4 Computing neighborhood graph...")
        n_pcs = min(n_components - 5, 50) if n_components > 5 else n_components
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)

        print("2.5 UMAP embedding (3D)...")
        sc.tl.umap(adata, n_components=3, min_dist=0.1, spread=1.0, random_state=42)

        print(f"2.6 {algorithm.capitalize()} clustering...")
        if algorithm == "louvain":
            sc.tl.louvain(adata, resolution=0.5, flavor="taynaud")
            adata.obs["leiden"] = adata.obs["louvain"]   # mirror for downstream code
        elif algorithm == "leiden":
            sc.tl.leiden(adata, resolution=0.5)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm!r} (expected 'leiden' or 'louvain')")

        # The normalized/log1p matrix above is only used for HVG/PCA/UMAP/clustering.
        # Marker statistics use one row per prompt and the raw SAE codes (see marker_stats).
        print("2.7 🎯 DIFFERENTIAL EXPRESSION ANALYSIS (one row per prompt, raw SAE codes)...")
        pdata = marker_stats.prompt_level(adata)
        if pdata.n_obs < adata.n_obs:
            print(f"   Averaged {adata.n_obs} token observations into {pdata.n_obs} prompts")
        marker_stats.rank_markers(pdata, n_genes=100)
        self.prompt_adata = pdata
        adata.uns['rank_genes_groups'] = pdata.uns['rank_genes_groups']
        print("   ✅ Differential expression analysis complete!")

        print(f"\nSaving complete AnnData object...")
        os.makedirs(processed_data_dir, exist_ok=True)
        adata.write(os.path.join(processed_data_dir, 'neural_atlas_complete.h5ad'))
        pdata.write(os.path.join(processed_data_dir, 'neural_atlas_prompt_level.h5ad'))

        pca_df = pd.DataFrame(
            adata.obsm['X_pca'],
            index=adata.obs.index,
            columns=[f'PC{i+1}' for i in range(adata.obsm['X_pca'].shape[1])]
        )
        pca_df.to_csv(os.path.join(processed_data_dir, 'pca_coordinates.csv'))

        umap_df = pd.DataFrame(
            adata.obsm['X_umap'],
            index=adata.obs.index,
            columns=['UMAP1', 'UMAP2', 'UMAP3']
        )
        umap_df.to_csv(os.path.join(processed_data_dir, 'umap_coordinates.csv'))

        clustering_df = pd.DataFrame({
            'prompt_id': adata.obs.index,
            'category': adata.obs['category'],
            'leiden_cluster': adata.obs['leiden']
        })
        clustering_df.to_csv(os.path.join(processed_data_dir, 'clustering_results.csv'), index=False)

        self.results['step2_scanpy_analysis'] = {
            'n_highly_variable_features': n_hvg,
            'n_pca_components': int(n_components),
            'n_leiden_clusters': int(len(adata.obs['leiden'].unique())),
            'differential_expression_completed': True
        }

        print("✅ Step 2 complete - Scanpy analysis saved")
        return adata

    def extract_and_save_marker_neurons(self, adata, processed_data_dir, results_dir):
        """
        Extract marker features (SAE) and save.
        """
        print(f"\nSTEP 3: EXTRACTING MARKER FEATURES (SAE, layer={self.sae_layer_idx})")
        print("-" * 60)

        marker_neurons = {}
        categories = adata.obs['category'].unique()
        pdata = self.prompt_adata
        thresholds = marker_stats.DEFAULT_THRESHOLDS

        print(f"STRICT Quality thresholds (one row per prompt, raw SAE codes):")
        print(f"  Adjusted p-value < {thresholds['max_pval_adj']}")
        print(f"  Score > {thresholds['min_score']}")
        print(f"  Log2 fold change > {thresholds['min_logfc']}")
        print(f"  Effect size (Cohen's d) > {thresholds['min_effect_size']}")

        for category in categories:
            print(f"\n🔍 {category.upper()} SPECIALIZED FEATURES:")
            try:
                top_markers = marker_stats.select_markers(pdata, category, thresholds)

                if len(top_markers) > 0:
                    marker_neurons[category] = top_markers
                    print(f"   ✅ Found {len(top_markers)} significant features")
                    print(f"   🥇 Top feature: {top_markers.iloc[0]['names']}")
                    print(f"      Score: {top_markers.iloc[0]['scores']:.3f}")
                    print(f"      P-value: {top_markers.iloc[0]['pvals_adj']:.2e}")
                    print(f"      Log FC: {top_markers.iloc[0]['logfoldchanges']:.3f}")
                else:
                    print(f"   ❌ No significant features found")
                    marker_neurons[category] = pd.DataFrame()

            except Exception as e:
                print(f"   ❌ Error processing {category}: {e}")
                marker_neurons[category] = pd.DataFrame()

        print(f"\nSaving marker features...")
        os.makedirs(results_dir, exist_ok=True)
        for category, markers in marker_neurons.items():
            if len(markers) > 0:
                filename = f"top_markers_{category}_saeL{self.sae_layer_idx}_{self.max_prompts_per_category}_prompts.csv"
                filepath = os.path.join(results_dir, filename)
                markers.to_csv(filepath, index=False)
                print(f"   ✅ {filename}: {len(markers)} features")

        with open(os.path.join(processed_data_dir, 'all_marker_neurons.pkl'), 'wb') as f:
            pickle.dump(marker_neurons, f)

        summary = {}
        for category, markers in marker_neurons.items():
            if len(markers) > 0:
                top_neuron = markers.iloc[0]
                summary[category] = {
                    'top_feature': top_neuron['names'],
                    'score': float(top_neuron['scores']),
                    'pvalue': float(top_neuron['pvals_adj']),
                    'logfc': float(top_neuron['logfoldchanges']),
                    'total_significant': len(markers)
                }

        summary_path = os.path.join(results_dir, f"marker_features_summary_saeL{self.sae_layer_idx}_{self.max_prompts_per_category}_prompts.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.results['step3_marker_neurons'] = {
            'categories_analyzed': list(categories),
            'marker_features_found': {cat: len(markers) for cat, markers in marker_neurons.items()},
            'summary_file': summary_path
        }

        print("✅ Step 3 complete - Marker features saved")
        return marker_neurons

    def compute_effect_size(self, adata, feature_name, category):
        """Cohen's d on prompt-level raw SAE codes (same space as the Wilcoxon test)."""
        pdata = marker_stats.prompt_level(adata) if 'n_tokens' not in adata.obs else adata
        if feature_name not in pdata.var_names:
            return 0.0
        col = pdata.var_names.get_loc(feature_name)
        in_group = (pdata.obs['category'].astype(str) == category).values
        return float(marker_stats.group_stats(pdata.X[:, [col]], in_group)[3][0])

    def validate_category_separation(self, adata, marker_neurons, min_neurons_per_category=1, seed=0):
        """Split-half replication: select markers on half of each category's prompts and
        test them on the held-out half (see marker_stats.split_half_validation)."""
        print(f"\n🔬 VALIDATING CATEGORY SEPARATION (split-half replication, seed={seed})")
        print("-" * 45)

        validation_results = marker_stats.split_half_validation(
            self.prompt_adata, min_markers=min_neurons_per_category, seed=seed
        )
        for category, r in validation_results.items():
            if r['valid']:
                print(f"   ✅ {category}: {r['n_replicated']}/{r['n_discovery_markers']} discovery markers replicate "
                      f"on {r['n_heldout_prompts']} held-out prompts ({r['replication_rate']:.0%})")
            else:
                print(f"   ❌ {category}: {r['reason']}")
        balance = next(iter(validation_results.values()))['balance_ratio'] if validation_results else 1.0
        if balance < 0.3:
            print(f"   ⚠️  IMBALANCED CATEGORIES: min/max prompt count ratio {balance:.2f}")
        return validation_results

    def permutation_test(self, adata, n_permutations=100, alpha=0.05, seed=0):
        """Shuffles category labels across prompts (not tokens)."""
        print(f"\n🎲 PERMUTATION TESTING ({n_permutations} permutations, prompt-level labels)")
        print("-" * 50)
        result = marker_stats.permutation_test(self.prompt_adata, n_permutations=n_permutations, alpha=alpha, seed=seed)
        for category, p_value in result['permutation_pvalues'].items():
            status = "✅ SIGNIFICANT" if p_value < alpha else "❌ NOT SIGNIFICANT"
            print(f"   {category}: {result['original_counts'][category]} markers, p={p_value:.3f} {status}")
        print(f"\n📊 PERMUTATION TEST SUMMARY:")
        print(f"   Significant categories: {result['n_significant_categories']}/{len(result['permutation_pvalues'])}")
        print(f"   Overall result: {'✅ SIGNIFICANT' if result['overall_significant'] else '❌ NOT SIGNIFICANT'}")
        return result

    def create_comprehensive_visualizations(self, adata, marker_neurons, figures_dir):
        print(f"\nSTEP 4: CREATING COMPREHENSIVE VISUALIZATIONS (SAE layer={self.sae_layer_idx})")
        print("-" * 65)

        plt.style.use('default')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 12

        # 4.1 UMAP
        print("4.1 Creating UMAP visualization...")
        reordered_indices = []
        for category in self.consistent_category_order:
            mask = adata.obs['category'] == category
            if mask.any():
                reordered_indices.extend(np.where(mask)[0])
        umap_adata = adata[reordered_indices, :].copy()
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(umap_adata, color='category', ax=ax, show=False, frameon=False)
        plt.title(f'UMAP by Category\n{self.model_name.upper()} (SAE layer {self.sae_layer_idx}, {self.max_prompts_per_category}/cat)')
        os.makedirs(figures_dir, exist_ok=True)
        umap_path = os.path.join(figures_dir, f'umap_categories_saeL{self.sae_layer_idx}_{self.max_prompts_per_category}prompts')
        plt.savefig(f'{umap_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{umap_path}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        print("4.2 Skipping Leiden plot...")

        # 4.3 Marker heatmap (top-K per category)
        print("4.3 Creating marker feature heatmap (top-K)...")
        top_k = 10
        selected = []
        for category, markers in marker_neurons.items():
            if len(markers) > 0:
                selected.extend(markers.head(top_k)['names'].tolist())
        selected = [n for n in selected if n in adata.var.index]
        if selected:
            tmp = adata[:, selected].copy()
            reordered_indices = []
            for category in self.consistent_category_order:
                mask = tmp.obs['category'] == category
                if mask.any():
                    reordered_indices.extend(np.where(mask)[0])
            tmp = tmp[reordered_indices, :].copy()
            fig, ax = plt.subplots(figsize=(15, 8))
            sc.pl.heatmap(tmp, selected, groupby='category', show=False, cmap='RdYlBu_r', standard_scale='var')
            heatmap_path = os.path.join(figures_dir, f'marker_features_heatmap_saeL{self.sae_layer_idx}_{self.max_prompts_per_category}prompts')
            plt.title('Marker Features Heatmap (SAE)')
            plt.savefig(f'{heatmap_path}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{heatmap_path}.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {heatmap_path}.pdf")
        else:
            print("   ⚠️  No valid features to plot")

        # 4.3.1 Complete marker heatmap (all significant)
        print("4.3.1 Creating complete marker features heatmap (all significant)...")
        all_markers = []
        for category, markers in marker_neurons.items():
            if len(markers) > 0:
                all_markers.extend(markers['names'].tolist())
        all_markers = [n for n in all_markers if n in adata.var.index]
        if all_markers:
            tmp = adata[:, all_markers].copy()
            reordered_indices = []
            for category in self.consistent_category_order:
                mask = tmp.obs['category'] == category
                if mask.any():
                    reordered_indices.extend(np.where(mask)[0])
            tmp = tmp[reordered_indices, :].copy()
            fig, ax = plt.subplots(figsize=(20, 12))
            sc.pl.heatmap(tmp, all_markers, groupby='category', show=False, cmap='RdYlBu_r', standard_scale='var')
            full_heatmap_path = os.path.join(figures_dir, f'complete_marker_features_heatmap_saeL{self.sae_layer_idx}_{self.max_prompts_per_category}prompts')
            plt.title('Complete Marker Features Heatmap (SAE)')
            plt.savefig(f'{full_heatmap_path}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{full_heatmap_path}.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {full_heatmap_path}.pdf")
        else:
            print("   ⚠️  No significant features across categories")

        # 4.3.2 Complete features heatmap (sampled if huge)
        print("4.3.2 Creating complete features heatmap (sampled if needed)...")
        all_feats = adata.var.index.tolist()
        if len(all_feats) > 1000:
            print(f"   ⚠️  Too many features ({len(all_feats)}), sampling 1000 for visualization")
            all_feats = list(np.random.choice(all_feats, 1000, replace=False))
        tmp = adata[:, all_feats].copy()
        fig, ax = plt.subplots(figsize=(20, 12))
        sc.pl.heatmap(tmp, all_feats, groupby='category', show=False, cmap='RdYlBu_r', standard_scale='var')
        all_heatmap_path = os.path.join(figures_dir, f'complete_features_heatmap_saeL{self.sae_layer_idx}_{self.max_prompts_per_category}prompts')
        plt.title('Complete Features Heatmap (SAE)')
        plt.savefig(f'{all_heatmap_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{all_heatmap_path}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        self.results['step4_visualizations'] = {
            'umap_categories': f'{umap_path}.pdf',
            'marker_heatmap': f'{heatmap_path}.pdf' if 'heatmap_path' in locals() else None,
            'complete_marker_heatmap': f'{full_heatmap_path}.pdf' if 'full_heatmap_path' in locals() else None,
            'complete_features_heatmap': f'{all_heatmap_path}.pdf' if 'all_heatmap_path' in locals() else None,
        }

        print("✅ Step 4 complete - Visualizations saved")

    def save_final_results(self, processed_data_dir):
        print(f"\nSTEP 5: SAVING FINAL RESULTS")
        print("-" * 40)
        results_path = os.path.join(processed_data_dir, 'complete_analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✅ Complete results saved: {results_path}")

        print(f"\n" + "=" * 80)
        print("ANALYSIS COMPLETE - ALL RESULTS SAVED")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"SAE layer: {self.sae_layer_idx} | d_sae={self.d_sae}")
        print(f"Prompts per category: {self.max_prompts_per_category}")
        print(f"Total observations: {self.results['step1_data_extraction']['total_observations']:,}")
        print(f"Total SAE features: {self.results['step1_data_extraction']['total_features']:,}")
        if 'step3_marker_neurons' in self.results:
            print(f"\nMarker features per category:")
            for cat, count in self.results['step3_marker_neurons']['marker_features_found'].items():
                print(f"  {cat}: {count} features")
        print(f"\nFiles saved under: {processed_data_dir}")
        return self.results

    def run_complete_pipeline(self, prompts_dir, results_dir):
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_results_dir = os.path.join(results_dir, timestamp)
        
        # Use timestamped directory for all outputs
        results_dir = timestamped_results_dir
        processed_data_dir = self.processed_data_dir  # Use the separate processed data directory
        figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(processed_data_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        print(f"\n📁 Results will be saved to timestamped directory:")
        print(f"   {results_dir}")
        print()

        activation_matrix, prompt_metadata, neuron_metadata = self.load_prompts_and_extract_activations(
            prompts_dir, processed_data_dir
        )
        adata = self.run_scanpy_analysis(
            activation_matrix, prompt_metadata, neuron_metadata, processed_data_dir
        )
        marker_neurons = self.extract_and_save_marker_neurons(
            adata, processed_data_dir, results_dir
        )

        # Held-out validation
        min_neurons_per_category = 1
        validation_results = self.validate_category_separation(
            adata, marker_neurons, min_neurons_per_category
        )
        validation_path = os.path.join(results_dir, f"category_validation_saeL{self.sae_layer_idx}_{self.max_prompts_per_category}_prompts.json")
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        self.results.setdefault('step3_marker_neurons', {})['category_validation'] = validation_results

        print(f"\n📋 VALIDATION SUMMARY:")
        print("-" * 28)
        valid_cats = [c for c, r in validation_results.items() if r.get('valid', False)]
        invalid_cats = [c for c, r in validation_results.items() if not r.get('valid', False)]
        print(f"✅ Valid categories: {len(valid_cats)}/{len(validation_results)}")
        for c in valid_cats:
            r = validation_results[c]
            print(f"   {c}: {r['n_discovery_markers']} discovery markers, replication rate: {r['replication_rate']:.1%}")
        if invalid_cats:
            print(f"❌ Invalid categories: {len(invalid_cats)}/{len(validation_results)}")
            for c in invalid_cats:
                print(f"   {c}: {validation_results[c].get('reason', 'Unknown reason')}")

        if self.run_permutation_test:
            perm = self.permutation_test(adata, self.n_permutations)
            perm_path = os.path.join(results_dir, f"permutation_test_saeL{self.sae_layer_idx}_{self.max_prompts_per_category}_prompts.json")
            with open(perm_path, 'w') as f:
                json.dump(perm, f, indent=2)
            self.results.setdefault('step3_marker_neurons', {})['permutation_results'] = perm

        self.create_comprehensive_visualizations(adata, marker_neurons, figures_dir)
        final_results = self.save_final_results(processed_data_dir)
        return final_results, adata, marker_neurons