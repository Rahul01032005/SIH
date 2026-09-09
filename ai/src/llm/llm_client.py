from abc import ABC, abstractmethod
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...
class UnconfiguredLLM(LLMClient):
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            include_reasoning=False,
        )

        return response.choices[0].message.content or ""

class GroqLLM(LLMClient):
    def __init__(self, model: str | None = None):
        key = os.getenv("GROQ_API_KEY")

        if not key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        from groq import Groq

        self.client = Groq(api_key=key)
        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b"
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_completion_tokens=500,
            reasoning_effort="low",
            include_reasoning=False,
        )

        content = response.choices[0].message.content

        if content:
            return content

        return "The model did not return a final answer."

def configured_llm() -> LLMClient | None:
    return GroqLLM() if os.getenv("GROQ_API_KEY") else None
