"""Type definitions for LLM interactions."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


@dataclass
class Card:
    """A card in The Mind game.

    In The Mind, cards are numbered from 1 to 100, and players must play them
    in ascending order without communicating.
    """

    number: int

    def __post_init__(self) -> None:
        """Validate card number is in valid range."""
        if not 1 <= self.number <= 100:
            raise ValueError("Card number must be between 1 and 100")


@dataclass
class Tool:
    """Base class for tools that can be used by the LLM."""

    name: str = ""
    description: str = ""
    parameters: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Set default parameters if none provided."""
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ToolSpec:
    """Specification for a tool that can be used by the LLM."""

    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    required_params: list[str]


@dataclass
class ToolCall:
    """A call to a tool by the LLM."""

    tool: str
    parameters: dict[str, Any]


@dataclass
class ToolResponse:
    """Response from a tool execution."""

    success: bool
    result: Optional[Any]
    error: Optional[str] = None


class Model(str, Enum):
    """Available LLM models."""

    # OpenAI models
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT35 = "gpt-3.5-turbo"

    # Anthropic models
    CLAUDE3_OPUS = "claude-3-opus-20240229"
    CLAUDE3_SONNET = "claude-3-sonnet-20240229"

    # Google models
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"


class Role(str, Enum):
    """Message roles in a conversation."""

    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Message:
    """A message in a conversation."""

    role: Role
    content: str
    tool_calls: Optional[list[ToolCall]] = None
    tool_response: Optional[ToolResponse] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to API-compatible dictionary.

        Returns:
            Dict with role and content, and optional tool information
        """
        return {
            "role": self.role.value,
            "content": self.content,
        }


@dataclass
class Response:
    """Encapsulates a response from the LLM API."""

    content: Optional[str]
    raw_response: Optional[Any]
    success: bool
    error: Optional[str]
    tool_calls: Optional[list[ToolCall]] = None

    def __repr__(self) -> str:
        if not self.success:
            return f"Response(error='{self.error}')"
        if self.content is None:
            return "Response(content=None)"
        return f"Response(content='{self.content[:50]}...')"

    def display(self, width: int = 50) -> None:
        """Display the response in a formatted way."""
        print("\nModel Output:")
        print("=" * width)
        if not self.success:
            print(f"Error: {self.error}")
        else:
            if self.content is not None:
                print(self.content)
            if self.tool_calls:
                print("\nTool Calls:")
                for call in self.tool_calls:
                    print(f"- {call.tool}: {call.parameters}")
        print("=" * width)


@dataclass
class PromptComponent:
    """A component of a prompt template."""

    role: Role
    dynamic_content: str

    def fill_dynamic_content(self, dynamic_content: dict[str, Any]) -> dict[str, Any]:
        """Fill dynamic parts of the message content.

        Args:
            dynamic_content: Values to fill placeholders in static content

        Returns:
            Complete message with filled content
        """
        filled_content = self.dynamic_content.format(**dynamic_content)
        return {
            "role": self.role.value,
            "content": filled_content,
        }


@dataclass
class PromptTemplate:
    """A template for generating prompts."""

    name: str
    version: str
    components: list[PromptComponent]
    temperature: float
    top_p: float
    model: Model
    available_tools: Optional[list[ToolSpec]] = None

    def __repr__(self) -> str:
        tools_len = len(self.available_tools) if self.available_tools is not None else 0
        return (
            f"PromptTemplate(name={self.name}, version={self.version}, "
            f"temperature={self.temperature}, top_p={self.top_p}, "
            f"model={self.model.value}, components={len(self.components)}, "
            f"tools={tools_len})"
        )

    def construct_prompt(self, dynamic_content: dict[str, Any]) -> list[dict[str, Any]]:
        """Construct the full prompt with dynamic content.

        Args:
            dynamic_content: Values to fill placeholders in components

        Returns:
            List of messages ready for API consumption
        """
        messages = [component.fill_dynamic_content(dynamic_content) for component in self.components]

        return messages
