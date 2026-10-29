# InterpAgent

AI Agent framework for automated interpretability workflows, with support for multiple LLM backends (OpenAI, Anthropic, Google Gemini) and a configurable workspace of tools and sub-agents.

## Requirements

- **Python** 3.10+
- **Workspace**: A `workspace/` directory with a `config.yaml` that defines your agent graph (see `workspace/config.yaml` in this repo).

## Quick Start

### 1. Clone and enter the repo

```bash
git clone https://github.com/YOUR_ORG/InterpAgent.git
cd InterpAgent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root with the API keys for the providers you want to use:

```env
# At least one of these is required, depending on which LLM you select in the app
OPENAI_API_KEY=your_openai_key
OPENAI_API_BASE=https://api.openai.com/v1   # optional, for custom endpoints

ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_BASE_URL=https://api.anthropic.com   # optional, for custom endpoint

GOOGLE_API_KEY=your_google_genai_key
GOOGLE_BASE_URL=   # optional, for custom endpoint
```

Do not commit `.env`; it is listed in `.gitignore`.

### SAE activation hook and scoring flags

Both agents read SAE inputs from the Gemma-2 MLP output **after** `post_feedforward_layernorm`
(TransformerLens `blocks.L.hook_mlp_out`, the representation the GemmaScope MLP dictionaries were
trained on) and exclude special tokens (BOS/EOS/pad) from per-text maxima and from the discovery
activation matrix. Two environment variables control this:

```bash
# default: post_feedforward_layernorm; set to legacy_mlp only to reproduce historical runs
export INTERPAGENT_SAE_HOOK=post_feedforward_layernorm
# default: 0 (special tokens excluded); set to 1 to include them
export INTERPAGENT_SCORE_SPECIAL_TOKENS=0
```

The code accepts both the old `sae-lens` API (`SAE.from_pretrained` returning a tuple) and the
current one (returning the SAE object).

### 5. Run the app

From the project root:

```bash
python main.py
```

The app will start and open in your browser (default: http://localhost:8501). Enter your name, choose an LLM, and start chatting.

## Project structure

```
InterpAgent/
├── app.py              # Streamlit UI entry (run via main.py)
├── main.py             # CLI launcher; use this to start the app
├── style.css           # App styling
├── requirements.txt    # Python dependencies
├── environment.yaml    # Conda environment (optional)
├── core/               # Agent runtime and UI utilities
│   ├── base/           # LLM wrappers, state, tools, planner
│   ├── st_utils/       # Streamlit sidebar and message rendering
│   └── generate.py     # Builds agent from workspace config
└── workspace/          # Agent config and tools (required)
    ├── config.yaml     # Defines agent_path and agent_graph
    ├── Agents/         # Agent modules and tools
    ├── prompts/        # Example prompts
    ├── data/
    └── results/
```

## Citation
Please cite as:
```
Marin-Llobet and Ferrando, "Automated Interpretability and Feature Discovery in Language Models with Agents", Preprint at arXiv https://arxiv.org/abs/2605.01555, 2026.
```

BibTeX Citation:
```
@article{marin2026interpagent,
  title     = {Automated Interpretability and Feature Discovery in Language Models with Agents},
  author    = {Marin-Llobet, Arnau and Ferrando, Javier},
  journal   = {arXiv preprint arXiv:2605.01555},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.01555}
}
```
