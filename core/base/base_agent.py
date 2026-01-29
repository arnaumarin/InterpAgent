# base_agent.py

from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.state import Command
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from typing import Literal
import inspect
import os

from .tool_node import CustomToolNode
from .state import AgentState
from .utils import import_function_from_file, import_module_from_file, extract_content_from_message
from .llm_models import get_llm
from .python_repl_tool import Py_tool
from .planner import AgentPlanner

class BaseAgent(Runnable):
    def __init__(self, model_name, log_fn, python_repl, supervisor_id, mode="supervised"):
        self.chat_history = []
        self.log = log_fn
        self.python_repl = python_repl
        self.supervisor_id = supervisor_id
        self.model_name = model_name
        self.mode = mode  # "supervised" or "autonomous"

        self.model = get_llm(model_name)
        self.agent_id = None
        self.prompt = None
        self.base_tools = []
        self.subagent_tools = []
        self.tool_node = None
        self.graph = None
        self.planner = None  # Will be initialized in build()

    def set_prompt(self, prompt_template, **format_kwargs):
        # Only call .format() if kwargs are provided; otherwise prompts with literal {word} would raise KeyError
        content = prompt_template.format(**format_kwargs) if format_kwargs else prompt_template
        self.prompt = SystemMessage(content=content)
        self.chat_history.append(self.prompt)

    def set_tools(self, tools_module):
        self.base_tools = [
            getattr(tools_module, name)
            for name in dir(tools_module)
            if isinstance(getattr(tools_module, name), BaseTool)
        ]
        python_repl_tool = Py_tool(self.python_repl).create()
        self.base_tools.append(python_repl_tool)

    def set_subagents(self, subagents: dict[str, Runnable]):
        """
        Register subagents and prepare tool-wrapping.
        Extends toolset.
        """
        for name, agent in subagents.items():
            self.add_subagent_tool(name, agent)

    def build(self):
        """
        Finalize tool_node and graph. Must be called after prompt/tools/subagents are set.
        """
        all_tools = self.base_tools + self.subagent_tools
        self.model = self.model.bind_tools(all_tools)
        self.tool_node = CustomToolNode(all_tools, self.log, self.agent_id)
        self.graph = self.compile_graph()
        
        # Initialize planner for the root agent only
        if self.agent_id == "🧠 Interp Agent":
            self.planner = AgentPlanner(self.model, mode=self.mode)

    def add_subagent_tool(self, name, agent, description=""):
        def make_tool(agent, name):
            def func(command: str):
                """Delegated subagent call with command or response
                Parameters:
                    - command: Your command or response to subagent.
                """
                response = agent.invoke([HumanMessage(content=command)])
                return "\n".join(
                    [extract_content_from_message(msg) for msg in response["messages"]]
                )
            func.__name__ = name
            return func, agent.agent_id
        subagent_func, name = make_tool(agent, name)
        subagent_tool = tool(subagent_func)
        subagent_tool.metadata = {'subagent_id':name}
        self.subagent_tools.append(subagent_tool)

    def invoke(self, messages):
        config = {"recursion_limit":150}
        return self.graph.invoke({"messages": messages}, config=config)

    def call(self, state:AgentState, config:RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
        self.log()
        new_messages = state["messages"]
        history_len = len(self.chat_history)
        self.chat_history.extend(new_messages)

        response = self.model.invoke(self.chat_history)

        remove_ids = []
        for i, msg in enumerate(new_messages):
            if check_if_img(msg):
                remove_ids.append(i)
        
        for i in remove_ids:
            rm = self.chat_history.pop(history_len+i)

        self.chat_history.append(response)

        if response.tool_calls:
            self.log(
                message=response, 
                sender=self.agent_id, 
                receiver=None, 
                m_type="reflect"
            )
            return Command(update={"messages": [response]}, goto="tool_node")
        else:
            self.log(
                message=response, 
                sender=self.agent_id, 
                receiver=self.supervisor_id, 
                m_type="report"
            )
            return Command(update={"messages": [response]}, goto="__end__")

    def compile_graph(self):
        graph = StateGraph(AgentState)
        graph.add_edge(START, "call_node")
        graph.add_node("call_node", self.call)
        graph.add_node("tool_node", self.tool_node)
        graph.add_edge("tool_node", "call_node")
        return graph.compile()
    
    def set_prompt_from_path(self, rel_path, py_file='prompt', promt_name='prompt_template'):
        prompt_template = import_function_from_file(rel_path, py_file, promt_name)
        self.set_prompt(prompt_template)

    def set_tool_from_path(self, rel_path, py_file='tools'):
        tools_module = import_module_from_file(rel_path, py_file)
        self.set_tools(tools_module)

    def get_agent_info(self, rel_path, py_file='agent'):
        agent_id = import_function_from_file(rel_path, py_file, 'agent_id')
        agent_description = import_function_from_file(rel_path, py_file, 'agent_description')
        return agent_id, agent_description
    
    def build_on_top(self, agent_name, agent_graph, rel_path):
        agent_path = os.path.join(rel_path,agent_name)
        agent_id, agent_description = self.get_agent_info(agent_path)
        self.agent_id = agent_id
        self.set_prompt_from_path(agent_path)
        self.set_tool_from_path(agent_path)
        for subagent_name, sub_graph in agent_graph.items():
            # Subagents inherit the mode from parent agent
            subagent = BaseAgent(self.model_name, self.log, self.python_repl, self.agent_id, mode=self.mode)
            subagent.build_on_top(subagent_name, sub_graph, rel_path)
            #self.subagents[subagent_name] = subagent # delete later
            self.add_subagent_tool(subagent_name, subagent)
        self.build()
            
        
def check_if_img(msg):
    if msg.type == 'human':
        if isinstance(msg.content, list):
            for cont in msg.content:
                if cont.get('text'):
                    if cont['text'] == "Here are the visual results. Assess the quality of the output or describe it within the context.":
                        return True
    
    return False

