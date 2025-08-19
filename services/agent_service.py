"""
Agent Service untuk mengelola kedua agent.
"""
from typing import Optional, Dict, Any
from agents import GeneralAgent, MarketingAgent


class AgentService:
    """Service untuk mengelola dan menggunakan kedua agent."""
    
    def __init__(self):
        """Initialize AgentService."""
        self.agents = {}
    
    def get_agent(self, agent_type: str, model_type: str = "openai"):
        """
        Get agent berdasarkan tipe.
        
        Args:
            agent_type: 'general' atau 'marketing'
            model_type: 'openai' atau 'gemini'
            
        Returns:
            Agent instance
        """
        agent_key = f"{agent_type}_{model_type}"
        
        if agent_key not in self.agents:
            if agent_type == "general":
                self.agents[agent_key] = GeneralAgent(model_type=model_type)
            elif agent_type == "marketing":
                self.agents[agent_key] = MarketingAgent(model_type=model_type)
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")
        
        return self.agents[agent_key]
    
    def chat(self, 
             query: str, 
             agent_type: str = "general", 
             model_type: str = "openai", 
             context: Optional[str] = None) -> str:
        """
        Chat dengan agent yang dipilih.
        
        Args:
            query: Pertanyaan user
            agent_type: 'general' atau 'marketing'
            model_type: 'openai' atau 'gemini'
            context: Konteks tambahan
            
        Returns:
            Response dari agent
        """
        try:
            agent = self.get_agent(agent_type, model_type)
            return agent.generate_response(query, context)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def add_marketing_knowledge(self, 
                               documents: list, 
                               metadata_list: list = None, 
                               model_type: str = "openai") -> bool:
        """
        Menambahkan dokumen ke marketing agent knowledge base.
        
        Args:
            documents: List dokumen text
            metadata_list: List metadata
            model_type: Tipe model yang digunakan
            
        Returns:
            True jika berhasil
        """
        try:
            agent = self.get_agent("marketing", model_type)
            return agent.add_marketing_documents(documents, metadata_list)
        except Exception as e:
            print(f"Error adding marketing knowledge: {str(e)}")
            return False
    
    def get_available_agents(self) -> Dict[str, Dict]:
        """
        Get daftar agent yang tersedia.
        
        Returns:
            Dict informasi agent
        """
        return {
            "general": {
                "name": "General Agent",
                "description": "Assistant umum seperti ChatGPT, bisa menjawab berbagai topik",
                "capabilities": [
                    "Teknologi dan programming",
                    "Sains dan penelitian", 
                    "Bisnis dan ekonomi",
                    "Pendidikan dan pembelajaran",
                    "Kesehatan dan lifestyle",
                    "Percakapan umum"
                ]
            },
            "marketing": {
                "name": "Marketing Analysis Agent", 
                "description": "Specialist untuk analisis marketing berbasis RAG",
                "capabilities": [
                    "Analisis tren pasar",
                    "Analisis kompetitor", 
                    "Segmentasi pelanggan",
                    "Strategi penjualan",
                    "Evaluasi kampanye",
                    "ROI dan marketing metrics"
                ]
            }
        }
    
    def get_available_models(self) -> Dict[str, Dict]:
        """
        Get daftar model yang tersedia.
        
        Returns:
            Dict informasi model
        """
        return {
            "openai": {
                "name": "OpenAI GPT-4O Mini",
                "description": "Model OpenAI yang powerful dan cepat",
                "best_for": "Analisis mendalam, reasoning kompleks"
            },
            "gemini": {
                "name": "Google Gemini 1.5 Flash", 
                "description": "Model Google yang cepat dan efisien",
                "best_for": "Response cepat, percakapan natural"
            }
        }
