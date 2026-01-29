import os
import datetime
import argparse
import json

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.messages import messages_from_dict, messages_to_dict


from core.st_utils import sidebar_utils, render_message, LoggerContext
from core.generate import generate_agent
from core.base.llm_models import list_available_llms

# Parse CLI --workspace argument
@st.dialog("👋 Welcome!", width="small")
def name_modal():
    st.markdown("### Please enter your name to continue:")
    name = st.text_input("Your name", value="User", placeholder="e.g., Alice", key="name_input")
    model_name = st.selectbox("Choose LLM:",list_available_llms(), key="model_input")

    if st.button("Start Chat"):
        if name.strip():
            st.session_state.user_name = name.strip()
            st.session_state.model_name = model_name
            st.session_state.agent_mode = "supervised"  # Always supervised; autonomous not exposed in UI
            st.rerun()
        else:
            st.warning("Please enter a valid name.")

def main():
    st.session_state.workspace_path = "workspace"

    # Get user name
    if "user_name" not in st.session_state:
        name_modal()
        st.stop()

    st.set_page_config(page_title="Interp Agent", layout="wide")
    
    st.title(f"InterpAgent powered by {st.session_state.model_name}")

    # Apply style.css
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    with st.sidebar:
        sidebar_utils()
            


    # Create supervisor agent if not loaded
    if "agent" not in st.session_state:
        # Logger
        st.session_state.logger = LoggerContext()
        # Agent initiate
        st.session_state.agent = generate_agent(
            st.session_state.workspace_path, 
            st.session_state.logger.log, 
            st.session_state.user_name, 
            st.session_state.model_name,
            st.session_state.agent_mode
        )

        # Render message history

    if "bokeh_stream" not in st.session_state:
        st.session_state.bokeh_stream = False

    for msg in st.session_state["logs"]:
        render_message(msg)

        # User input
    user_input = st.chat_input("Type your message here...")

    # Handle user message
    if user_input:
        user_msg = HumanMessage(content=user_input)
        st.session_state.logger.log(
            message=user_msg, 
            sender=st.session_state.agent.supervisor_id, 
            receiver=st.session_state.agent.agent_id,
            m_type = 'human'
        )

        response = st.session_state.agent.invoke([user_msg])
    
    if st.session_state.bokeh_stream:
        import streamlit.components.v1 as components
        bokeh_url = "http://localhost:5007/"
        components.html(f'<iframe src="{bokeh_url}" width="100%" height="1000" style="border:none;"></iframe>', height=1000)
        st.session_state.bokeh_stream=False

if __name__ == "__main__":
    main()
