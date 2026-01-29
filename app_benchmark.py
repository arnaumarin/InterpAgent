import os
import sys
import datetime
import json
import streamlit as st

from langchain_core.messages import HumanMessage, messages_to_dict, ChatMessage

from core.generate import generate_agent
from core.st_utils import LoggerContext

REPEAT = str(sys.argv[1])
CURRENT_ITER = str(sys.argv[2])
MODEL_NAME = str(sys.argv[3])


#PROMPT = "Hi"
# Raw data path is outside agentic-work: set RAW_DATA_PATH env or replace below placeholder.
_raw_path = os.environ.get("RAW_DATA_PATH", "path/to/raw/neuropixel_data/AL031_2019-12-02")
PROMPT = f"""Run the whole Interp agent pipeline END-TO-END, without asking any feedback to user. you must answer to your subagents by yourself. Do not ask user for confirmation. Here are the parameters that might be useful during spikesorting phase. Raw data is located at "{_raw_path}" which is neuropixel recording, save at "data/processed_AL31", sorter: 'kilosort4'. Curation method:  vlm curation(unit 2 is good, unit 9,12 is bad), without merging units. When plotting unit features, plot only first 10 units."""
WORKSPACE_PATH = "workspace"
USER_NAME = "benchmark"

if "user_name" not in st.session_state:
    st.session_state.user_name = USER_NAME
if "model_name" not in st.session_state:
    st.session_state.model_name = MODEL_NAME
if "workspace_path" not in st.session_state:
    st.session_state.workspace_path = WORKSPACE_PATH
if "logger" not in st.session_state:
    st.session_state.logger = LoggerContext()
if "agent" not in st.session_state:
    st.session_state.agent = generate_agent(
        st.session_state.workspace_path, 
        st.session_state.logger.log, 
        st.session_state.user_name, 
        st.session_state.model_name
    )


st.title("🔁 Benchmark Runner")
st.markdown(f"**Model:** `{st.session_state.model_name}` &nbsp;&nbsp;&nbsp; **Iteration:** `{CURRENT_ITER}/{REPEAT}`")

try:
    user_msg = HumanMessage(content=PROMPT)
    st.session_state.logger.log(
        message=user_msg,
        sender=st.session_state.agent.supervisor_id,
        receiver=st.session_state.agent.agent_id,
        m_type="human"
    )
    response = st.session_state.agent.invoke([user_msg])

except Exception as e:
    if isinstance(e, Exception):
        exception = " ".join(e.args)
    else:
        exception = str(e)
    st.session_state.logger.log(
        message=ChatMessage(content=exception, role="error"),
        sender="system",
        receiver=st.session_state.user_name,
        m_type="error"
    )

log_folder = os.path.join(WORKSPACE_PATH, "logs_benchmark")
os.makedirs(log_folder, exist_ok=True)
model_folder = os.path.join(log_folder, st.session_state.model_name)
os.makedirs(model_folder, exist_ok=True)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join(model_folder, f'{CURRENT_ITER}_{timestamp}.json')

serialized = [
    {**msg, "message": messages_to_dict([msg["message"]])[0]}
    for msg in st.session_state.get("logs", [])
]
with open(log_file, "w", encoding="utf-8") as f:
    json.dump(serialized, f, indent=2)


st.success(f"✅ Log saved to `{log_file}`")
import psutil
pid = os.getpid()
p = psutil.Process(pid)
p.terminate()