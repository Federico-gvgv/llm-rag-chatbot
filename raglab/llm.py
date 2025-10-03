# LLM interface for different providers

import os, requests
from typing import Optional
from openai import OpenAI

class LLM:
    def __init__(self, provider: str, model: str, temperature=0.2, max_tokens=512, timeout_sec=60):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        if provider == "openai":
            self.client = OpenAI()
        else:
            self.client = None # Placeholder for other providers

    # Generate text based on the prompt
    def generate(self, prompt: str) -> str:
        if self.provider == "ollama": # Ollama API call
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434") # default host
            response = requests.post( # POST request to Ollama
                f"{host}/api/generate", # endpoint
                json={  # request body
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature}},
                timeout=self.timeout_sec, # timeout
            )
            response.raise_for_status() # raise error for bad status
            return response.json().get("response", "") # return response text
        elif self.provider == "openai": # OpenAI API call
            response = self.client.chat.completions.create( # chat completion
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content # return response text
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
                