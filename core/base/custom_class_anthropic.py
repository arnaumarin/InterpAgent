import copy
import re
import warnings
import json
import requests
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)
import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, model_validator
from typing_extensions import NotRequired
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, retry_if_result

# Placeholder imports for langchain_core classes and methods.
# Adjust imports as needed based on your environment.
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    from_env,
    get_pydantic_field_names,
    secret_from_env,
)
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_anthropic.output_parsers import extract_tool_calls
_message_type_lookups = {
    "human": "user",
    "ai": "assistant",
}


def _format_image(image_url: str) -> Dict:
    regex = r"^data:(?P<mime_type>image/.+);base64,(?P<data>.+)$"
    match = re.match(regex, image_url)
    if match is None:
        raise ValueError(
            "Anthropic custom endpoint only supports base64-encoded images currently."
        )
    return {
        "type": "base64",
        "media_type": match.group("mime_type"),
        "data": match.group("data"),
    }


def _merge_messages(
    messages: Sequence[BaseMessage],
) -> List[Union[SystemMessage, AIMessage, HumanMessage]]:
    """Merge runs of human/tool messages into single human messages with content blocks."""
    merged: list = []
    for curr in messages:
        if isinstance(curr, ToolMessage):
            # Convert tool messages into human messages with tool results
            if (
                isinstance(curr.content, list)
                and curr.content
                and all(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in curr.content
                )
            ):
                curr = HumanMessage(curr.content)  # type: ignore[misc]
            else:
                curr = HumanMessage(  # type: ignore[misc]
                    [
                        {
                            "type": "tool_result",
                            "content": curr.content,
                            "tool_use_id": curr.tool_call_id,
                            "is_error": curr.status == "error",
                        }
                    ]
                )
        last = merged[-1] if merged else None
        if any(
            all(isinstance(m, c) for m in (curr, last))
            for c in (SystemMessage, HumanMessage)
        ):
            if isinstance(last.content, str):
                new_content: List = [
                    {"type": "text", "text": last.content}
                ]
            else:
                new_content = copy.copy(cast(list, last.content))
            if isinstance(curr.content, str):
                new_content.append({"type": "text", "text": curr.content})
            else:
                new_content.extend(curr.content)
            merged[-1] = curr.model_copy(update={"content": new_content})
        else:
            merged.append(curr)
    return merged


def _format_messages(
    messages: List[BaseMessage],
) -> Tuple[Union[str, List[Dict], None], List[Dict]]:
    """Format messages for anthropic."""

    """
    [
                {
                    "role": _message_type_lookups[m.type],
                    "content": [_AnthropicMessageContent(text=m.content).model_dump()],
                }
                for m in messages
            ]
    """
    system: Union[str, List[Dict], None] = None
    formatted_messages: List[Dict] = []

    merged_messages = _merge_messages(messages)
    for i, message in enumerate(merged_messages):
        if message.type == "system":
            if system is not None:
                raise ValueError("Received multiple non-consecutive system messages.")
            elif isinstance(message.content, list):
                system = [
                    (
                        block
                        if isinstance(block, dict)
                        else {"type": "text", "text": block}
                    )
                    for block in message.content
                ]
            else:
                system = message.content
            continue

        role = _message_type_lookups[message.type]
        content: Union[str, List]

        if not isinstance(message.content, str):
            # parse as dict
            assert isinstance(
                message.content, list
            ), "Anthropic message content must be str or list of dicts"

            # populate content
            content = []
            for block in message.content:
                if isinstance(block, str):
                    content.append({"type": "text", "text": block})
                elif isinstance(block, dict):
                    if "type" not in block:
                        raise ValueError("Dict content block must have a type key")
                    elif block["type"] == "image_url":
                        # convert format
                        source = _format_image(block["image_url"]["url"])
                        content.append({"type": "image", "source": source})
                    elif block["type"] == "tool_use":
                        # If a tool_call with the same id as a tool_use content block
                        # exists, the tool_call is preferred.
                        if isinstance(message, AIMessage) and block["id"] in [
                            tc["id"] for tc in message.tool_calls
                        ]:
                            overlapping = [
                                tc
                                for tc in message.tool_calls
                                if tc["id"] == block["id"]
                            ]
                            content.extend(
                                _lc_tool_calls_to_anthropic_tool_use_blocks(overlapping)
                            )
                        else:
                            block.pop("text", None)
                            content.append(block)
                    elif block["type"] == "text":
                        text = block.get("text", "")
                        # Only add non-empty strings for now as empty ones are not
                        # accepted.
                        # https://github.com/anthropics/anthropic-sdk-python/issues/461
                        if text.strip():
                            content.append(
                                {
                                    k: v
                                    for k, v in block.items()
                                    if k in ("type", "text", "cache_control")
                                }
                            )
                    elif block["type"] == "tool_result":
                        tool_content = _format_messages(
                            [HumanMessage(block["content"])]
                        )[1][0]["content"]
                        content.append({**block, **{"content": tool_content}})
                    else:
                        content.append(block)
                else:
                    raise ValueError(
                        f"Content blocks must be str or dict, instead was: "
                        f"{type(block)}"
                    )
        else:
            content = message.content

        # Ensure all tool_calls have a tool_use content block
        if isinstance(message, AIMessage) and message.tool_calls:
            content = content or []
            content = (
                [{"type": "text", "text": message.content}]
                if isinstance(content, str) and content
                else content
            )
            tool_use_ids = [
                cast(dict, block)["id"]
                for block in content
                if cast(dict, block)["type"] == "tool_use"
            ]
            missing_tool_calls = [
                tc for tc in message.tool_calls if tc["id"] not in tool_use_ids
            ]
            cast(list, content).extend(
                _lc_tool_calls_to_anthropic_tool_use_blocks(missing_tool_calls)
            )

        formatted_messages.append({"role": role, "content": content})
        #print('This is system',system)
    return system, formatted_messages

class _AnthropicToolUse(TypedDict):
    type: Literal["tool_use"]
    name: str
    input: dict
    id: str

def _lc_tool_calls_to_anthropic_tool_use_blocks(
    tool_calls: List[ToolCall],
) -> List[_AnthropicToolUse]:
    blocks = []
    for tool_call in tool_calls:
        blocks.append(
            _AnthropicToolUse(
                type="tool_use",
                name=tool_call["name"],
                input=tool_call["args"],
                id=cast(str, tool_call["id"]),
            )
        )
    return blocks

class AnthropicTool(TypedDict):
    """Anthropic tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    cache_control: NotRequired[Dict[str, str]]


def convert_to_anthropic_tool(
    tool: Union[Dict[str, Any], Type, Callable, BaseTool],
) -> AnthropicTool:
    oai_formatted = convert_to_openai_tool(tool)["function"]
    anthropic_formatted = AnthropicTool(
        name=oai_formatted["name"],
        description=oai_formatted["description"],
        input_schema=oai_formatted["parameters"],
    )
    return anthropic_formatted



def _create_usage_metadata() -> UsageMetadata:
    # If custom endpoint doesn't provide usage metadata, we can return empty.
    return UsageMetadata(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0
    )


class ChatAnthropic_H(BaseChatModel):
    """A modified ChatAnthropic class for a custom Anthropic-compatible endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    model: str = Field(alias="model_name")
    max_tokens: int = Field(default=1024, alias="max_tokens_to_sample")
    temperature: Optional[float] = 0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    default_request_timeout: Optional[float] = Field(None, alias="timeout")
    max_retries: int = 2
    stop_sequences: Optional[List[str]] = Field(None, alias="stop")
    anthropic_base_url: str = ""
    anthropic_api_key: str = Field(default_factory=from_env(["ANTHROPIC_API_KEY"], default=""))
    default_headers: Optional[Mapping[str, str]] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    streaming: bool = False
    stream_usage: bool = False  # Not supported here

    @property
    def _llm_type(self) -> str:
        return "anthropic-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        # If you have secrets, return them here
        return {}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain", "chat_models", "anthropic"]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "model_kwargs": self.model_kwargs,
            "streaming": self.streaming,
            "max_retries": self.max_retries,
            "default_request_timeout": self.default_request_timeout,
        }

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict) -> Any:
        all_required_field_names = get_pydantic_field_names(cls)
        # This function used to build model kwargs. If needed, adapt as required.
        return values

    @model_validator(mode="after")
    def post_init(self):
        # Setup headers
        self._headers = {
            "Content-Type": "application/json",
            "x-api-key": self.anthropic_api_key,
        }
        if self.default_headers:
            self._headers.update(self.default_headers)
        return self

    def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: Optional[List[str]] = None,
            **kwargs: Dict,
    ) -> Dict:
        messages = self._convert_input(input_).to_messages()
        system, formatted_messages = _format_messages(messages)
        
        # Construct the base body dictionary with required fields
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
        }
        
        # Add optional fields only if they have non-None values
        if self.temperature is not None:
            body["temperature"] = self.temperature
        
        if system:
            body["system"] = system
            
        # Add any additional kwargs to body if they have non-None values
        body.update({k: v for k, v in kwargs.items() if v is not None})
        
        # Construct the complete payload
        # Handle model ID format - don't add :0 if it's already there
        model_id = f"anthropic.{self.model}"
        if not model_id.endswith(":0"):
            model_id += ":0"
            
        payload = {
            "modelId": model_id,
            "contentType": "application/json",
            "accept": "application/json",
            "body": body
        }
        
        # Remove any top-level None values before returning
        return {k: v for k, v in payload.items() if v is not None}

    def _format_output(self, data: Any, **kwargs: Any) -> ChatResult:
        data_dict = data
        
        # Handle different response formats from custom endpoint
        if "content" in data_dict:
            content = data_dict["content"]
        elif "body" in data_dict and "content" in data_dict["body"]:
            # If response is wrapped in body
            content = data_dict["body"]["content"]
        elif isinstance(data_dict, dict) and "choices" in data_dict:
            # OpenAI-style response
            content = data_dict["choices"][0]["message"]["content"]
        else:
            # Try to extract content from the response
            if isinstance(data_dict, str):
                content = [{"type": "text", "text": data_dict}]
            else:
                raise ValueError(f"Unexpected response format: {type(data_dict)}, keys: {list(data_dict.keys()) if isinstance(data_dict, dict) else 'N/A'}")
        llm_output = {
            k: v for k, v in data_dict.items() if k not in ("content", "role", "type")
        }
        if len(content) == 1 and content[0]["type"] == "text":
            #print("This is zl content",content)
            msg = AIMessage(content=content[0]["text"])
        elif any(block["type"] == "tool_use" for block in content):
            tool_calls = extract_tool_calls(content)
            msg = AIMessage(
                content=content,
                tool_calls=tool_calls,
            )
        else:
            msg = AIMessage(content=content)
        #msg.usage_metadata = _create_usage_metadata(data['usage'])
        return ChatResult(
            generations=[ChatGeneration(message=msg)],
            llm_output=llm_output,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        base_url = self.anthropic_base_url.rstrip("/") if self.anthropic_base_url else ""
        if not base_url:
            raise ValueError("anthropic_base_url must be set for ChatAnthropic_H.")
        response = requests.post(base_url + "/messages", headers=self._headers, json=payload)
        
        if response.status_code != 200:
            print(f"ERROR: HTTP {response.status_code} - {response.text}")
            response.raise_for_status()
            
        response_data = response.json()
        return self._format_output(response_data)
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        @retry(
            stop=stop_after_attempt(3), # Retry up to 3 times
            wait=wait_fixed(2), # Wait 2 seconds between retries
            retry=( # Retry on RequestError OR specific HTTP status codes
                retry_if_exception_type(httpx.RequestError) | 
                retry_if_result(lambda r: r.status_code in [429, 500, 502, 503, 504])
            )
        )
        async def _make_api_call_with_retry(client: httpx.AsyncClient, url: str, headers: dict, json_payload: dict) -> httpx.Response:
            print(f"Attempting API call to {url}...") # Add log for attempt
            response = await client.post(url, headers=headers, json=json_payload)
            print(f"API call attempt finished with status: {response.status_code}") # Add log for result
            # Note: We don't raise_for_status here, retry condition handles status codes
            return response
        
        try:
            # Prepare payload and headers
            payload = self._get_request_payload(messages, stop=stop, **kwargs)
            #print("Payload prepared:", json.dumps(payload, indent=4))
            #print("Headers prepared:", self._headers)

            # Make the HTTP request using the retry helper
            base_url = (self.anthropic_base_url.rstrip("/") + "/messages") if self.anthropic_base_url else ""
            if not base_url:
                raise ValueError("anthropic_base_url must be set for ChatAnthropic_H.")
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await _make_api_call_with_retry(
                    client,
                    base_url,
                    self._headers,
                    payload
                )

                # Log response details
                #print("Response status code:", response.status_code)
                #print("Response headers:", response.headers)
                #print("Response text:", response.text)

                # Validate response AFTER retries if necessary (optional, depends on desired handling)
                # If the response status is still bad after retries, the JSON decode will likely fail 
                # or you might want to raise an explicit error here based on status_code.
                # For now, we let the JSON decode handle it.
                # response.raise_for_status() # Example: Raise if still an error after retries

                # Process the response (JSON decoding with existing error logging)
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    print(f"ERROR: Failed to decode JSON response from API even after retries. Status Code: {response.status_code}")
                    print(f"Raw Response Text: {response.text}")
                    raise e # Re-raise the exception after logging

                # Format the output
                chat_result = self._format_output(response_data, **kwargs)
                return chat_result

        except httpx.RequestError as exc:
            print(f"HTTP request error: {exc}")
            raise

        except Exception as exc:
            print(f"An unexpected error occurred: {exc}")
            raise


    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[Dict[str, str], Literal["any", "auto"], str]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        # Still implemented, but may not work as expected with custom endpoint
        formatted_tools = [convert_to_anthropic_tool(tool) for tool in tools]
        if not tool_choice:
            pass
        elif isinstance(tool_choice, dict):
            kwargs["tool_choice"] = tool_choice
        elif isinstance(tool_choice, str) and tool_choice in ("any", "auto"):
            kwargs["tool_choice"] = {"type": tool_choice}
        elif isinstance(tool_choice, str):
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        else:
            raise ValueError(
                f"Unrecognized 'tool_choice' type {tool_choice=}. Expected dict, "
                f"str, or None."
            )
        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        tool_name = convert_to_anthropic_tool(schema)["name"]
        llm = self.bind_tools([schema], tool_choice=tool_name)
        if isinstance(schema, type) and is_basemodel_subclass(schema):
            output_parser: OutputParserLike = PydanticToolsParser(tools=[schema], first_tool_only=True)
        else:
            output_parser = JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks([parser_none], exception_key="parsing_error")
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    def get_num_tokens_from_messages(
        self,
        messages: List[BaseMessage],
        tools: Optional[
            Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]]
        ] = None,
    ) -> int:
        # Not supported by custom endpoint
        # Return a dummy value or implement logic if supported
        return 0