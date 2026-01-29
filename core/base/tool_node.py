from langchain_core.runnables import Runnable
from langchain_core.messages import ToolMessage, HumanMessage, ChatMessage
from langchain_core.runnables.config import RunnableConfig
import json
# --------------------
# Tool Node with logging
# --------------------
class CustomToolNode(Runnable):
    def __init__(self, tools, log_fn, agent_id):
        self.tools = {t.name: t for t in tools}
        self.log = log_fn
        self.agent_id = agent_id

    def invoke(self, state, config:RunnableConfig):
        tool_calls = state["messages"][-1].tool_calls
        messages = []
        for call in tool_calls:
            name = call["name"]
            args = call["args"]
            tool_call_id = call['id']
            tool = self.tools.get(name)
            if tool:
                response_format =  getattr(tool, "response_format", None)
                metadata = getattr(tool, "metadata")
                try:
                    if response_format == "content_and_artifact":
                        
                        self.log(
                            message=ChatMessage(content=args['query'], role="code"), 
                            sender=self.agent_id, 
                            receiver=name, 
                            m_type="tool_input"
                        )
                        result = tool.func(**args)
                        content, encoded_imgs = result
                        tool_msg = ToolMessage(
                            tool_call_id=tool_call_id,
                            content=content
                        )
                        messages.append(tool_msg)
                        self.log(
                            message=tool_msg, 
                            sender=name, 
                            receiver=self.agent_id, 
                            m_type="tool_output"
                        )
                        if encoded_imgs:
                            img_msg = HumanMessage(
                                [{"type": "text", "text": "Here are the visual results. Assess the quality of the output or describe it within the context."}]+
                                [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in encoded_imgs]
                            )
                            messages.append(img_msg)   
                            self.log(
                                message=img_msg, 
                                sender=name, 
                                receiver=self.agent_id, 
                                m_type="reflect_img"
                            )
                    else:
                        if metadata:
                            subagent_id = metadata.get("subagent_id")
                            
                            self.log(
                                message=ChatMessage(content=args["command"], role="command"), 
                                sender=self.agent_id, 
                                receiver=subagent_id, 
                                m_type="assign"
                            )
                            result = tool.func(**args)
                            tool_msg = ToolMessage(
                                tool_call_id=tool_call_id,
                                content=result
                            )
                            messages.append(tool_msg)
                            # self.log(
                            #     message=tool_msg, 
                            #     sender=self.agent_id, 
                            #     receiver=subagent_id, 
                            #     m_type="assign"
                            # )
                        else:
                            self.log(
                                message=ChatMessage(content="\n".join(f"{k}: {v}" for k, v in args.items()), role="code"), 
                                sender=self.agent_id, 
                                receiver=name, 
                                m_type="tool_input"
                            )
                            result = tool.func(**args)
                            tool_msg = ToolMessage(
                                tool_call_id=tool_call_id,
                                content=result
                            )
                            messages.append(tool_msg)
                            self.log(
                                message=tool_msg, 
                                sender=name, 
                                receiver=self.agent_id, 
                                m_type="tool_output"
                            )
                except Exception as e:
                    tool_msg = ToolMessage(content=f"Tool `{name}` failed: {e}", tool_call_id=tool_call_id)
                    self.log(
                        message=tool_msg, 
                        sender=name, 
                        receiver=self.agent_id, 
                        m_type="tool_output"
                    )
                    messages.append(tool_msg)
            else:
                tool_msg = ToolMessage(content=f"[Unknown tool: {name}]", tool_call_id=tool_call_id)
                self.log(
                    message=tool_msg, 
                    sender=name, 
                    receiver=self.agent_id, 
                    m_type="tool_output"
                )
                messages.append(tool_msg)
        return {"messages": messages}


