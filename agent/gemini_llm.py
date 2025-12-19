import requests
import google.generativeai as genai
from langchain_core.language_models import LLM
from typing import Optional, List
from config.settings import settings
import time
import json

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)

class HybridLLM(LLM):
    """Hybrid LLM: OpenRouter primary, Gemini fallback"""

    model_name: str = settings.OPENROUTER_MODEL
    gemini_model: str = settings.DEVELOPMENT_MODEL
    max_retries: int = 3
    retry_delays: list = [2, 4, 8]  # Progressive delays

    @property
    def _llm_type(self) -> str:
        return "hybrid"

    def _call_openrouter(self, prompt: str) -> Optional[str]:
        """Call OpenRouter API"""
        try:
            headers = {
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 4000,
                "top_p": 0.9
            }
            
            response = requests.post(
                settings.OPENROUTER_API_URL,
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"🟡 OpenRouter error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"🟡 OpenRouter request failed: {e}")
            return None

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini API as fallback"""
        try:
            model = genai.GenerativeModel(self.gemini_model)
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
                return response.text
            return None
                
        except Exception as e:
            print(f"🟡 Gemini request failed: {e}")
            return None

    def _call_with_retry(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call with retry logic: try OpenRouter first, fallback to Gemini"""
        
        # Try OpenRouter
        for attempt in range(self.max_retries):
            try:
                print(f"🔵 Calling OpenRouter (attempt {attempt + 1}/{self.max_retries})...")
                result = self._call_openrouter(prompt)
                if result:
                    print(f"✅ OpenRouter success on attempt {attempt + 1}")
                    return result
                    
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt])
                    
            except Exception as e:
                print(f"OpenRouter error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt])
        
        # Fallback to Gemini
        print(f"🔴 OpenRouter failed, trying Gemini...")
        for attempt in range(self.max_retries):
            try:
                print(f"🔴 Calling Gemini (attempt {attempt + 1}/{self.max_retries})...")
                result = self._call_gemini(prompt)
                if result:
                    print(f"✅ Gemini success on attempt {attempt + 1}")
                    return result
                    
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt])
                    
            except Exception as e:
                print(f"Gemini error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt])
        
        return "[Error] Both OpenRouter and Gemini failed"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Main call method"""
        return self._call_with_retry(prompt, stop)

# Create singleton instance
hybrid_llm = HybridLLM()

# Keep gemini_llm for backwards compatibility
gemini_llm = hybrid_llm