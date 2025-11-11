import google.generativeai as genai
from langchain_core.language_models import LLM
from typing import Optional, List
from config.settings import settings
import time
import asyncio

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)

class GeminiLLM(LLM):
    """Gemini LLM wrapper with retry logic"""

    model_name: str = settings.DEVELOPMENT_MODEL
    max_retries: int = 3
    retry_delays: list = [1, 2, 4]  # Progressive delays

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call_with_retry(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call Gemini with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # print(f"🔴 Calling Gemini API (attempt {attempt + 1}/{self.max_retries})...")
                print(f"🔴")


                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=4000,
                        top_p=0.9,
                        top_k=40
                    )
                )

                if response.text and response.text.strip():
                    print(f"Gemini API call successful on attempt {attempt + 1}")
                    return response.text
                else:
                    print(f"Gemini returned empty response on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delays[attempt])
                        continue
                    return "[Gemini Error] Empty response after all retries"

            except Exception as e:
                print(f"Gemini API error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt])
                    continue
                return f"[Gemini Error] Failed after {self.max_retries} attempts: {e}"

        return "[Gemini Error] Max retries exceeded"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Main call method with retries"""
        return self._call_with_retry(prompt, stop)

# Create singleton instance
gemini_llm = GeminiLLM()