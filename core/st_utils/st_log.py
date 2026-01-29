# util.py

import os
import json
import datetime
import textwrap
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

class LoggerContext:
    def __init__(self):
        if "logs" not in st.session_state:
            st.session_state.logs = []

    def log(self, message=None, sender=None, receiver=None, m_type=None):
        if message is not None:
            message_log = {"message": message, "sender": sender, "receiver": receiver, "m_type": m_type}
            st.session_state["logs"].append(message_log)
            render_message(message_log)

def display_tool(message, sender, receiver, m_type):
    if m_type == "tool_output":
        with st.expander(f"{sender} -> {receiver}", expanded=False):
            st.write("**Tool Output:**")
            if message.content:
                st.code(message.content) 
            else:
                st.code("No output") 
    elif m_type == "tool_input":
        with st.expander(f"{sender} -> {receiver}", expanded=False):
            st.write("**Tool Input:**")
            if message.content:
                st.code(message.content) 
            else:
                st.code("No input") 

def display_img(message, sender, receiver, m_type):
    if m_type == 'reflect_img':
        role = "ai"
    elif m_type == 'human_img':
        role = "human"

    block_parts = [f"<div class='chat-block {role}'><div class='prefix'>{sender}</div>"]
    for i, content_dict in enumerate(message.content):
        if content_dict["type"] == "text":
            continue
            # block_parts.append(f"<div class='chat-bubble'>{content_dict['text']}</div>")
            # block_parts.append("</div>")
        elif content_dict["type"] == "image_url":
            encoded_img = content_dict.get("image_url", {}).get("url")
            if encoded_img:
                block_parts.append(f"<div class='chat-block {role}'>")
                img_html = f"""
                <img src="{encoded_img}" class="chat-image"><br>
                <a href="{encoded_img}" download="image_{i}.png" style="font-size:12px; color:blue;">⬇️ Download</a>
                """
                block_parts.append(img_html)
                block_parts.append("</div>")
    st.markdown("".join(block_parts),unsafe_allow_html=True)

def rephrase_ai_message(content):
    if isinstance(content,list):
        rephrased = []
        for i in content:
            if isinstance(i, dict):
                if "text" in i:
                    rephrased.append(i.get("text"))
            elif isinstance(i, str):
                rephrased.append(i)
        rephrased = "\n".join(rephrased)
    else:
        if isinstance(content, dict):
            if "text" in content:
                rephrased = content.get("text")
        elif isinstance(content, str):
            rephrased = content
        else:
            rephrased = "No messages"
    rephrased = (rephrased.replace("\\(", "")
            .replace("\\)", "")
            .replace("\\[", "")
            .replace("\\]", "")
            .replace("$$", "")
            .replace("$", ""))
    return rephrased

def render_message(message_log):
    """
    Renders conversation history from a list of messages, handling multiple tool calls.
    """

    message = message_log["message"]
    sender = message_log["sender"]
    receiver = message_log["receiver"]
    m_type = message_log["m_type"] # human(hm), report(ai), reflect(ai), assign(str), tool_input, tool_output, reflect_img, human_img,  hide

    if m_type == 'hide':
        return
    
    if m_type == 'tool_input' or m_type == 'tool_output':
        display_tool(message, sender, receiver, m_type)
        return

    elif m_type == 'reflect_img' or m_type == 'human_img':
        display_img(message, sender, receiver, m_type)
        return
    
    else:
        if m_type == "human":
            content = message.content
            prefix = sender
        elif m_type == "report":
            content = rephrase_ai_message(message.content)
            if receiver == st.session_state.user_name:
                prefix = sender
            else:
                prefix = f"{sender} -> {receiver}"
        elif m_type == "reflect":
            if not message.content:
                return
            content = rephrase_ai_message(message.content)
            if not content:
                return
            prefix = sender
        elif m_type == "assign":
            content = rephrase_ai_message(message.content)
            prefix = f"{sender} -> {receiver}"
        else:
            content = message.content
            prefix = f"{sender} -> {receiver}"

        role = "human" if m_type=="human" else "ai"

        html = f"""
        <div class="chat-block {role}">
            <div class="prefix">{prefix}</div>
            <div class="chat-bubble">{content}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)