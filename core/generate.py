import os
import yaml
from langchain_experimental.utilities.python import PythonREPL
from .base import BaseAgent

# Build supervisor agent from config.yaml in the workspace
def generate_agent(workspace_path, log_fn, user_name, model_name, mode="supervised"):
    # Load config
    config_path = os.path.join(workspace_path, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Parse config fields
    agent_rel_path = config.get("agent_path")
    agent_name = config.get("root_agent")
    agent_graph = config.get("agent_graph").get(agent_name)

    # Resolve agent directory
    agent_abs_path = os.path.join(workspace_path, agent_rel_path)

    # Create Python REPL environment
    python_repl = PythonREPL()

    # Initialize and build agent with mode
    agent = BaseAgent(model_name, log_fn, python_repl, user_name, mode=mode)
    agent.build_on_top(agent_name, agent_graph, agent_abs_path)

    return agent
