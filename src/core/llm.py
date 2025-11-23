"""LLM integration layer supporting multiple providers."""

import google.generativeai as genai
from typing import List, Dict, Any, Optional, AsyncGenerator
from enum import Enum

from src.core.config import settings
from src.core.logger import setup_logger

logger = setup_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(self, provider: LLMProvider = LLMProvider.GEMINI):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider to use
        """
        self.provider = provider
        self._setup_client()
        logger.info(f"LLM client initialized with provider: {provider}")

    def _setup_client(self):
        """Set up provider-specific client."""
        if self.provider == LLMProvider.GEMINI:
            genai.configure(api_key=settings.google_api_key)
            self.model = genai.GenerativeModel(settings.gemini_model)
            logger.info(f"Gemini model configured: {settings.gemini_model}")

        elif self.provider == LLMProvider.OPENAI:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            # TODO: Initialize OpenAI client
            logger.info("OpenAI client configured")

        elif self.provider == LLMProvider.CLAUDE:
            if not settings.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            # TODO: Initialize Anthropic client
            logger.info("Claude client configured")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate completion from prompt.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            if self.provider == LLMProvider.GEMINI:
                generation_config = {
                    "temperature": temperature,
                }
                if max_tokens:
                    generation_config["max_output_tokens"] = max_tokens

                # Combine system prompt and user prompt
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"

                response = await self.model.generate_content_async(
                    full_prompt,
                    generation_config=generation_config,
                )

                return response.text

            # TODO: Implement other providers
            else:
                raise NotImplementedError(f"Provider {self.provider} not implemented")

        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming completion.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Generation temperature

        Yields:
            Generated text chunks
        """
        try:
            if self.provider == LLMProvider.GEMINI:
                generation_config = {"temperature": temperature}

                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"

                response = await self.model.generate_content_async(
                    full_prompt,
                    generation_config=generation_config,
                    stream=True,
                )

                async for chunk in response:
                    if chunk.text:
                        yield chunk.text

            else:
                raise NotImplementedError(f"Provider {self.provider} not implemented")

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """
        Chat completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Generation temperature

        Returns:
            Assistant response
        """
        try:
            if self.provider == LLMProvider.GEMINI:
                # Convert messages to Gemini format
                chat = self.model.start_chat(history=[])

                # Add history (excluding the last user message)
                for msg in messages[:-1]:
                    if msg["role"] == "user":
                        chat.history.append(
                            {
                                "role": "user",
                                "parts": [msg["content"]],
                            }
                        )
                    elif msg["role"] == "assistant":
                        chat.history.append(
                            {
                                "role": "model",
                                "parts": [msg["content"]],
                            }
                        )

                # Send last user message
                response = chat.send_message(
                    messages[-1]["content"],
                    generation_config={"temperature": temperature},
                )

                return response.text

            else:
                raise NotImplementedError(f"Provider {self.provider} not implemented")

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if self.provider == LLMProvider.GEMINI:
            return self.model.count_tokens(text).total_tokens
        else:
            # Approximate for other providers
            return len(text.split())


# Global LLM client
_llm_client: Optional[LLMClient] = None


def get_llm_client(provider: LLMProvider = LLMProvider.GEMINI) -> LLMClient:
    """Get or create global LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(provider)
    return _llm_client
