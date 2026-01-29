import os
import datetime
import json

import streamlit as st
from langchain_core.messages import messages_from_dict, messages_to_dict

def sidebar_utils():
    st.markdown("## 🔧 Session Control")

    log_folder = os.path.join(st.session_state.workspace_path, "logs_benchmark/gpt-4o")
    os.makedirs(log_folder, exist_ok=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⚠️ Restart", use_container_width=True):
            st.toast("Session reset in progress...")
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("💾 Save", use_container_width=True):
            serialized = [
                {**msg, "message": messages_to_dict([msg["message"]])[0]}
                for msg in st.session_state.get("logs", [])
            ]
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            username = st.session_state.get("user_name", "user")
            model_name = st.session_state.get("model_name", "Unknown")
            log_file = os.path.join(log_folder, f'{model_name}_{username}_{timestamp}.json')

            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2)

            st.success(f"✅ Log saved to `{log_file}`")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📂 Load Conversation")
    with col2:
        load_clicked = st.button("📂 Load", use_container_width=True)

    files = [f for f in os.listdir(log_folder) if f.endswith(".json")]
    files = sorted(files,reverse=True)

    if files:
        selected_file = st.selectbox("Select file", files, label_visibility="collapsed")
        if load_clicked:
            file_path = os.path.join(log_folder, selected_file)
            with open(file_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
            st.session_state["logs"] = [
                {**msg, "message": messages_from_dict([msg["message"]])[0]}
                for msg in logs
            ]
            st.success(f"✅ Loaded `{selected_file}`")
            st.rerun()
    else:
        st.info("No log files found.")