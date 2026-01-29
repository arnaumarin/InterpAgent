from typing import TypedDict, List
from langgraph.graph.message import AnyMessage

class AgentState(TypedDict):
    messages: List[AnyMessage]