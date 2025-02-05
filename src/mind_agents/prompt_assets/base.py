"""Base interfaces and configuration for LLM integrations."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

from .types import PromptTemplate


class Message(BaseModel):
    """A message in a conversation."""

    role: str
    content: str


class LLMConfig(BaseModel):
    """Configuration for LLM models."""

    api_key: str
    model_endpoint: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None


class Response(BaseModel):
    """Structured response from LLM models."""

    content: str
    raw_response: dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


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
