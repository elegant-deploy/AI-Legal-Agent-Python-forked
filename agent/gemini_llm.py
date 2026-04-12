from openai import OpenAI
from langchain_core.language_models import LLM
from langchain_core.outputs import LLMResult
from typing import Optional, List
from config.settings import settings
import time
import asyncio

# Commented out Perplexity code
# client = OpenAI(api_key=settings.PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai/")

# class PerplexityLLM(LLM):
#     """Perplexity LLM wrapper with retry logic"""

#     # model_name: str = "sonar"
#     model_name: str = "sonar"  # Changed from "sonar" to smaller, faster model
#     max_retries: int = 3
#     retry_delays: list = [1, 2, 4]  # Progressive delays

#     def __init__(self):
#         super().__init__()
#         self.max_retries = 3
#         self.retry_delays = [1, 2, 4]
#         self.model_name = "sonar"

#     @property
#     def _llm_type(self) -> str:
#         return "perplexity"

#     def _call_with_retry(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         """Call Perplexity with retry logic"""
#         for attempt in range(self.max_retries):
#             try:
#                 print(f"🔴 Calling Perplexity API (attempt {attempt + 1}/{self.max_retries})...")

#                 response = client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=0.1,
#                     max_tokens=4000
#                 )

#                 if response.choices and response.choices[0].message.content:
#                     print(f" Perplexity API call successful on attempt {attempt + 1}")
#                     return response.choices[0].message.content
#                 else:
#                     print(f"Perplexity returned empty response on attempt {attempt + 1}")
#                     if attempt < self.max_retries - 1:
#                         time.sleep(self.retry_delays[attempt])
#                         continue
#                     return "[Perplexity Error] Empty response after all retries"

#             except Exception as e:
#                 print(f"Perplexity API error on attempt {attempt + 1}: {e}")
#                 if attempt < self.max_retries - 1:
#                         time.sleep(self.retry_delays[attempt])
#                         continue
#                 return f"[Perplexity Error] Failed after {self.max_retries} attempts: {e}"

#         return "[Perplexity Error] Max retries exceeded"

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         """Main call method with retries"""
#         return self._call_with_retry(prompt, stop)

# # Create singleton instance
# perplexity_llm = PerplexityLLM()


# Uncommented Gemini code
from google import genai
from google.genai import types

gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

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
                print(f"🔴 Calling Gemini API (attempt {attempt + 1}/{self.max_retries})...")

                response = gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=1024,
                        top_p=0.9,
                        top_k=40,
                    ),
                )

                if response.text and response.text.strip():
                    print(f" Gemini API call successful on attempt {attempt + 1}")
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

    def stream_content(self, prompt: str, start_time: Optional[float] = None):
        """Stream Gemini response in chunks. Yields str chunks. Logs TTFT when start_time given."""
        try:
            t0 = start_time if start_time is not None else time.time()
            print(f"[{time.perf_counter()-t0:.2f}s] 🔴 LLM: Sending request to Gemini (prompt {len(prompt)} chars)...")
            response = gemini_client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    top_p=0.9,
                    top_k=40,
                ),
            )
            first = True
            for chunk in response:
                if chunk.text and chunk.text.strip():
                    if first and start_time is not None:
                        print(f"[{time.perf_counter()-t0:.2f}s] 🔴 LLM: First token received (TTFT)")
                        first = False
                    yield chunk.text
        except Exception as e:
            yield f"\n\n[Stream error: {e}]"

# Create singleton instance
gemini_llm = GeminiLLM()


# Uncommented Perplexity code for fallback
client = OpenAI(api_key=settings.PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai/")

class PerplexityLLM(LLM):
    """Perplexity LLM wrapper with retry logic"""

    model_name: str = "sonar"
    max_retries: int = 3
    retry_delays: list = [1, 2, 4]  # Progressive delays

    def __init__(self):
        super().__init__()
        self.max_retries = 3
        self.retry_delays = [1, 2, 4]
        self.model_name = "sonar"

    @property
    def _llm_type(self) -> str:
        return "perplexity"

    def _call_with_retry(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call Perplexity with retry logic"""
        for attempt in range(self.max_retries):
            try:
                print(f"🔴 Calling Perplexity API (attempt {attempt + 1}/{self.max_retries})...")

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4000
                )

                if response.choices and response.choices[0].message.content:
                    print(f" Perplexity API call successful on attempt {attempt + 1}")
                    return response.choices[0].message.content
                else:
                    print(f"Perplexity returned empty response on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delays[attempt])
                        continue
                    return "[Perplexity Error] Empty response after all retries"

            except Exception as e:
                print(f"Perplexity API error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt])
                    continue
                return f"[Perplexity Error] Failed after {self.max_retries} attempts: {e}"

        return "[Perplexity Error] Max retries exceeded"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Main call method with retries"""
        return self._call_with_retry(prompt, stop)

# Create singleton instance
perplexity_llm = PerplexityLLM()
