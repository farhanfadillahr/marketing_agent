"""
Base Agent class untuk semua agent.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from config import settings


class BaseAgent(ABC):
    """Base class untuk semua agent."""

    def __init__(self, model_type: str = "telkom-ai"):
        """
        Initialize base agent.
        
        Args:
            model_type: Tipe model yang digunakan ('telkom-ai' atau 'gemini')
        """
        self.model_type = model_type
        self.settings = settings
    
    @abstractmethod
    def generate_response(self, query: str, context: Optional[str] = None, **kwargs) -> str:
        """
        Generate response dari agent.
        
        Args:
            query: Pertanyaan user
            context: Konteks tambahan jika ada
            **kwargs: Parameter tambahan
            
        Returns:
            Response dari agent
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Mendapatkan system prompt untuk agent.
        
        Returns:
            System prompt string
        """
        pass
    
    def _get_model_client(self):
        """Get client untuk model yang dipilih."""
        if self.model_type == "telkom-ai":
            from openai import OpenAI
            return OpenAI(
                api_key=self.settings.telkom_ai_api_key,
                base_url=self.settings.telkom_ai_base_url,
                default_headers={"x-api-key": self.settings.telkom_ai_api_key})
        elif self.model_type == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.settings.gemini_api_key)
            return genai.GenerativeModel(self.settings.gemini_model)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
