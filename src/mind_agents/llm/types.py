"""Type definitions for LLM interactions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional


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


class ReasoningEffort(str, Enum):
    """Reasoning effort for O1/O3 models."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ModelMetadata:
    """Metadata for a model."""

    provider: Literal["openai", "anthropic", "google", "groq"]
    model_id: str  # The actual model ID to use with the API
    context_length: int
    supports_system_messages: bool = True
    supports_temperature: bool = True
    supports_reasoning_effort: bool = False
    reasoning_effort: Optional[ReasoningEffort] = None  # For O3 models that support reasoning effort


# Model metadata mapping
MODELS: dict[str, ModelMetadata] = {
    # OpenAI models
    "GPT4": ModelMetadata(
        provider="openai",
        model_id="gpt-4",
        context_length=8192,
    ),
    "GPT4_TURBO": ModelMetadata(
        provider="openai",
        model_id="gpt-4-turbo",
        context_length=128000,
    ),
    "GPT35_TURBO": ModelMetadata(
        provider="openai",
        model_id="gpt-3.5-turbo",
        context_length=4096,
    ),
    "GPT_O1": ModelMetadata(
        provider="openai",
        model_id="o1-2024-12-17",
        context_length=4096,
        supports_system_messages=False,
        supports_temperature=False,
    ),
    "GPT_O3_MINI_LOW": ModelMetadata(
        provider="openai",
        model_id="o3-mini-2025-01-31",
        context_length=4096,
        supports_system_messages=False,
        supports_temperature=False,
        supports_reasoning_effort=True,
        reasoning_effort=ReasoningEffort.LOW,
    ),
    "GPT_O3_MINI_MED": ModelMetadata(
        provider="openai",
        model_id="o3-mini-2025-01-31",
        context_length=4096,
        supports_system_messages=False,
        supports_temperature=False,
        supports_reasoning_effort=True,
        reasoning_effort=ReasoningEffort.MEDIUM,
    ),
    "GPT_O3_MINI_HIGH": ModelMetadata(
        provider="openai",
        model_id="o3-mini-2025-01-31",
        context_length=4096,
        supports_system_messages=False,
        supports_temperature=False,
        supports_reasoning_effort=True,
        reasoning_effort=ReasoningEffort.HIGH,
    ),
    # Anthropic models
    "CLAUDE3_OPUS": ModelMetadata(
        provider="anthropic",
        model_id="claude-3-opus-20240229",
        context_length=200000,
    ),
    "CLAUDE3_SONNET": ModelMetadata(
        provider="anthropic",
        model_id="claude-3-sonnet-20240229",
        context_length=200000,
    ),
    "CLAUDE3_HAIKU": ModelMetadata(
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        context_length=200000,
    ),
    "CLAUDE35_SONNET": ModelMetadata(
        provider="anthropic",
        model_id="claude-3-5-sonnet-20240620",
        context_length=200000,
    ),
    "CLAUDE35_HAIKU": ModelMetadata(
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        context_length=200000,
    ),
    # Google models
    "GEMINI_1_5_FLASH": ModelMetadata(
        provider="google",
        model_id="gemini-1.5-flash",
        context_length=32768,
    ),
    "GEMINI_1_5_PRO": ModelMetadata(
        provider="google",
        model_id="gemini-1.5-pro",
        context_length=32768,
    ),
    "GEMINI_2_0_FLASH": ModelMetadata(
        provider="google",
        model_id="gemini-2.0-flash-001",
        context_length=32768,
    ),
    # Groq models
    # Alibaba Cloud models
    "QWEN_2.5_32B": ModelMetadata(
        provider="groq",
        model_id="qwen-2.5-32b",
        context_length=32768,
    ),
    "QWEN_2.5_CODER_32B": ModelMetadata(
        provider="groq",
        model_id="qwen-2.5-coder-32b",
        context_length=32768,
    ),
    # DeepSeek / Alibaba Cloud
    "DEEPSEEK_QWEN_32B": ModelMetadata(
        provider="groq",
        model_id="deepseek-r1-distill-qwen-32b",
        context_length=32768,
    ),
    # DeepSeek / Meta
    "DEEPSEEK_LLAMA_70B": ModelMetadata(
        provider="groq",
        model_id="deepseek-r1-distill-llama-70b",
        context_length=4096,
    ),
    # Google
    "GEMMA_9B": ModelMetadata(
        provider="groq",
        model_id="gemma2-9b",
        context_length=2048,
    ),
    # Meta
    "LLAMA_3.1_8B": ModelMetadata(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        context_length=4096,
    ),
    "LLAMA_3.2_11B_VISION": ModelMetadata(
        provider="groq",
        model_id="llama-3.2-11b-vision-preview",
        context_length=4096,
    ),
    "LLAMA_3.2_90B_VISION": ModelMetadata(
        provider="groq",
        model_id="llama-3.2-90b-vision-preview",
        context_length=4096,
    ),
    "LLAMA_3.3_70B": ModelMetadata(
        provider="groq",
        model_id="llama-3.3-70b-8192",
        context_length=8192,
    ),
    # Mistral AI
    "MISTRAL_8X7B": ModelMetadata(
        provider="groq",
        model_id="mixtral-8x7b-32768",
        context_length=32768,
    ),
}


@dataclass
class LLMConfig:
    """Configuration for LLM models."""

    api_key: str
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string")
        if not self.api_key:
            raise ValueError("api_key cannot be empty")

        if not isinstance(self.model_name, str):
            raise TypeError("model_name must be a string")
        if self.model_name not in MODELS:
            raise ValueError(f"Unknown model: {self.model_name}")

        if not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be a number")
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")

        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int):
                raise TypeError("max_tokens must be an integer")
            if self.max_tokens <= 0:
                raise ValueError("max_tokens must be positive")

        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise TypeError("system_prompt must be a string")

    @property
    def metadata(self) -> ModelMetadata:
        """Get metadata for this model."""
        return MODELS[self.model_name]


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


class Role(str, Enum):
    """Message roles in a conversation."""

    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    FUNCTION = "function"


def get_model_metadata(model_name: str) -> ModelMetadata:
    """Get metadata for a model.

    Args:
        model_name: The name of the model to get metadata for

    Returns:
        The model's metadata

    Raises:
        ValueError: If the model name is not found
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODELS[model_name]


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

    content: str
    raw_response: dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None

    def __repr__(self) -> str:
        if not self.success:
            return f"Response(error='{self.error}')"
        return f"Response(content='{self.content[:50]}...')"

    def display(self, width: int = 50) -> None:
        """Display the response in a formatted way."""
        print("\nModel Output:")
        print("=" * width)
        if not self.success:
            print(f"Error: {self.error}")
        else:
            print(self.content)
            if self.tool_calls:
                print("\nTool Calls:")
                for call in self.tool_calls:
                    print(f"- {call['tool']}: {call['parameters']}")
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
    model_name: str  # Changed from model: Model to use string
    available_tools: Optional[list[ToolSpec]] = None

    def __post_init__(self) -> None:
        """Validate template values."""
        if self.model_name not in MODELS:
            raise ValueError(f"Unknown model: {self.model_name}")

    def __repr__(self) -> str:
        tools_len = len(self.available_tools) if self.available_tools is not None else 0
        return (
            f"PromptTemplate(name={self.name}, version={self.version}, "
            f"temperature={self.temperature}, top_p={self.top_p}, "
            f"model={self.model_name}, components={len(self.components)}, "  # Changed from model.value to model_name
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


class LLM(ABC):
    """Base interface for LLM interactions."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the LLM with configuration.

        Args:
            config: Configuration for the LLM including API key and parameters
        """
        self.config = config

    @abstractmethod
    async def generate(
        self,
        template: PromptTemplate,
        dynamic_content: dict[str, Any],
    ) -> Response:
        """Generate a response from the LLM.

        Args:
            template: The prompt template containing components, tools, and configuration
            dynamic_content: Values to fill placeholders in components

        Returns:
            Response containing the generated text and metadata

        Raises:
            Exception: If the API call fails
        """
        pass

    @abstractmethod
    async def validate_api_key(self) -> bool:
        """Validate that the API key is correct and working.

        Returns:
            bool indicating if the API key is valid
        """
        pass
