"""
General Agent - Seperti ChatGPT yang bisa menjawab pertanyaan umum.
"""
from typing import Optional
from .base_agent import BaseAgent
from langchain_core.messages import HumanMessage, SystemMessage


class GeneralAgent(BaseAgent):
    """
    Agent General untuk menjawab pertanyaan umum seperti ChatGPT.
    Fleksibel, ramah, dan bisa menjawab berbagai topik.
    """
    
    def __init__(self, model_type: str = "openai"):
        super().__init__(model_type)
    
    def get_system_prompt(self) -> str:
        """System prompt untuk General Agent."""
        return """Anda adalah asisten AI yang ramah dan membantu, seperti ChatGPT. 
        
        Karakteristik Anda:
        - Ramah dan sopan dalam berinteraksi
        - Mampu menjawab berbagai topik: teknologi, sains, bisnis, pendidikan, kesehatan, dan percakapan umum
        - Memberikan jawaban yang informatif, akurat, dan mudah dipahami
        - Dapat membantu dengan berbagai tugas seperti menulis, analisis, perhitungan, dan brainstorming
        - Mengakui ketika tidak tahu jawaban dan menyarankan sumber yang bisa membantu
        - Menjaga percakapan tetap natural dan engaging
        
        Selalu berikan jawaban yang lengkap namun tidak berbelit-belit. Jika ada informasi yang kurang jelas, 
        jangan ragu untuk meminta klarifikasi dari user."""
    
    def generate_response(self, query: str, context: Optional[str] = None, **kwargs) -> str:
        """
        Generate response untuk pertanyaan umum.
        
        Args:
            query: Pertanyaan user
            context: Konteks tambahan (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Response dari General Agent
        """
        try:
            if self.model_type == "openai":
                client = self._get_model_client()
                
                messages = [
                    SystemMessage(content=self.get_system_prompt())
                ]
                
                if context:
                    messages.append(SystemMessage(content=f"Konteks tambahan: {context}"))
                
                messages.append(HumanMessage(content=query))
                
                response = client.invoke(messages)
                return response.content
                
            elif self.model_type == "gemini":
                client = self._get_model_client()
                
                # Untuk Gemini, gabungkan system prompt dengan query
                full_prompt = f"{self.get_system_prompt()}\n\n"
                
                if context:
                    full_prompt += f"Konteks tambahan: {context}\n\n"
                
                full_prompt += f"Pertanyaan: {query}\n\nJawaban:"
                
                response = client.generate_content(full_prompt)
                return response.text
                
        except Exception as e:
            return f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}"
