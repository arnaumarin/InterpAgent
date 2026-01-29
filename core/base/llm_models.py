import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from .custom_class_anthropic import ChatAnthropic_H
from .custom_class_gemini import ChatGoogleGenerativeAI_H

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_BASE_URL = os.getenv("GOOGLE_BASE_URL")

MAX_TOKEN=1024

# Define all possible models, mapping to factory functions
all_llms = {
    # OpenAI
    "gpt-4o": lambda token: ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        temperature=0.0,
        max_tokens=token
    ),
    # "gpt-4.1": lambda token: ChatOpenAI(
    #     model="gpt-4.1-2025-04-14",
    #     api_key=OPENAI_API_KEY,
    #     base_url=OPENAI_API_BASE,
    #     temperature=0.0,
    #     max_tokens=token
    # ),
    "o3": lambda token: ChatOpenAI(
        model="o3-2025-04-16",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        max_tokens=token
    ),
    # "o4-mini": lambda token: ChatOpenAI(
    #     model="o4-mini-2025-04-16",
    #     api_key=OPENAI_API_KEY,
    #     base_url=OPENAI_API_BASE,
    #     max_tokens=token
    # ),
    #"gpt-o3-mini": lambda: ChatOpenAI(model="o3-mini", api_key=OPENAI_API_KEY),

    # Claude
    "claude-4-sonnet": lambda token: ChatAnthropic_H(
        model="claude-sonnet-4-20250514-v1",
        anthropic_base_url=ANTHROPIC_BASE_URL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0.0,
        max_tokens=token,
    ),
    # "claude-4-opus": lambda token: ChatAnthropic_H(
    #     model="claude-opus-4-20250514-v1",
    #     anthropic_base_url=ANTHROPIC_BASE_URL,
    #     anthropic_api_key=ANTHROPIC_API_KEY,
    #     temperature=0.0,
    #     max_tokens=token,
    # ),
    "claude-3.7-sonnet": lambda token: ChatAnthropic_H(
        model="claude-3-7-sonnet-20250219-v1",
        anthropic_base_url=ANTHROPIC_BASE_URL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0.0,
        max_tokens=token,
    ),

    # Gemini
    "gemini-2.5-pro": lambda token: ChatGoogleGenerativeAI_H(
        model="gemini-2.5-pro",
        google_api_key=GOOGLE_API_KEY,
        client_options={"api_endpoint": GOOGLE_BASE_URL} if GOOGLE_BASE_URL else {},
        max_output_tokens=token,
        thinking_budget=None, 
        include_thoughts=False,
        temperature = 0.0,
        convert_system_message_to_human=True,
        ),
    "gemini-2.5-flash": lambda token: ChatGoogleGenerativeAI_H(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        client_options={"api_endpoint": GOOGLE_BASE_URL} if GOOGLE_BASE_URL else {},
        max_output_tokens=token,
        thinking_budget=None, 
        include_thoughts=False,
        temperature = 0.0,
        convert_system_message_to_human=True,
        )
}

#Filter based on which keys are present
def _key_available(model_name):
    if model_name.startswith("gpt") or model_name.startswith("o"):
        return bool(OPENAI_API_KEY)
    elif model_name.startswith("claude"):
        return bool(ANTHROPIC_API_KEY)
    elif model_name.startswith("gemini"):
        return bool(GOOGLE_API_KEY)
    return False


# Public: return list of available model names based on existing keys
def list_available_llms():
    return [name for name in all_llms if _key_available(name)]

# Public: return model instance if available
def get_llm(model_name: str, token=MAX_TOKEN):
    if model_name not in all_llms:
        raise ValueError(f"Model '{model_name}' not found.")
    if not _key_available(model_name):
        raise ValueError(f"API key for model '{model_name}' is not available.")
    return all_llms[model_name](token)
