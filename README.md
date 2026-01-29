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

For a full environment (including optional GPU/science stack), use Conda:

```bash
conda env create -f environment.yaml
conda activate deviceAgent
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