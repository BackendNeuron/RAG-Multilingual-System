# src/phase2/llm_integration.py

import os
from groq import Groq

class LLMClient:
    """
    Handles LLM API integration using Groq and context-aware prompt engineering.
    Uses LLaMA 3.3 70B Versatile model by default.
    """

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Args:
            api_key: Groq API key (from env var or parameter)
            model_name: LLM model to use (defaults to env var or 'llama-3.3-70b-versatile')
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key must be provided via parameter or GROQ_API_KEY env var.")

        self.model_name = model_name or os.environ.get("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
        self.client = Groq(api_key=self.api_key)

    def create_prompt(self, query: str, context: str = "", system_message: str = "You are a helpful assistant.") -> list:
        """
        Build chat messages by combining context and current query.
        Returns list of messages compatible with Groq API.
        """
        messages = []
        if context:
            messages.append({"role": "system", "content": system_message + "\n" + context})
        else:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        return messages

    def generate_answer(
        self,
        query: str,
        context: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> str:
        """
        Generate an answer from the LLM using Groq API.

        Args:
            query: User question
            context: Optional conversation or retrieved context
            max_tokens: Max output tokens (ignored by Groq for now)
            temperature: Randomness in output (ignored by Groq for now)
        """
        messages = self.create_prompt(query, context)

        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling LLM API: {e}"
