"""
Marketing Agent - Berbasis RAG untuk analisis marketing.
"""
from typing import Optional, List
from .base_agent import BaseAgent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from services.vector_service import VectorService
import google.generativeai as genai


class MarketingAgent(BaseAgent):
    """
    Agent Marketing Analysis berbasis RAG.
    Fokus khusus pada analisis marketing dengan knowledge base.
    """
    
    def __init__(self, model_type: str = "openai"):
        super().__init__(model_type)
        self.vector_service = VectorService(collection_name=self.settings.qdrant_marketing_collection)
        self.retrieval_qa = None
        self._setup_retrieval_qa()
    
    def _setup_retrieval_qa(self):
        """Setup RetrievalQA chain untuk RAG."""
        print("Setting up RetrievalQA for Marketing Agent...")
        print(f"Using model type: {self.model_type}")
        try:
            if self.model_type == "openai":
                llm = ChatOpenAI(
                    api_key=self.settings.openai_api_key,
                    model=self.settings.openai_model,
                    temperature=0.3  # Lower temperature untuk konsistensi
                )
            # elif self.model_type == "gemini":
            #     genai.configure(api_key=self.settings.gemini_api_key)
            #     llm = genai.GenerativeModel(self.settings.gemini_model)

                # Get retriever dari vector service
                retriever = self.vector_service.get_retriever(search_kwargs={"k": 3})
                
                if retriever:
                    self.retrieval_qa = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={
                            "prompt": self._get_rag_prompt_template()
                        }
                    )
        except Exception as e:
            print(f"Error setting up RetrievalQA: {str(e)}")
    
    def get_system_prompt_example(self) -> str:
        """System prompt untuk Marketing Agent."""
        return """Anda adalah Marketing Analysis Agent yang ahli dalam analisis pemasaran.

        Fokus keahlian Anda HANYA pada:
        - Analisis tren pasar dan industri
        - Analisis kompetitor dan positioning
        - Segmentasi dan targeting pelanggan
        - Strategi penjualan dan distribution
        - Evaluasi kampanye marketing
        - ROI dan metrics pemasaran
        - Consumer behavior dan market research
        - Digital marketing dan social media strategy
        - Pricing strategy dan value proposition
        - Brand management dan brand awareness

        ATURAN PENTING:
        1. HANYA menjawab pertanyaan yang berkaitan dengan marketing/pemasaran
        2. Jika pertanyaan di luar topik marketing, jawab: "Maaf, saya hanya dapat membantu dengan pertanyaan seputar analisis marketing dan pemasaran."
        3. Jika informasi tidak tersedia dalam knowledge base, jawab: "Informasi tersebut tidak tersedia dalam basis data marketing saya."
        4. Gunakan data dan insights dari knowledge base untuk memberikan analisis yang mendalam
        5. Berikan rekomendasi yang actionable berdasarkan data yang ada

        Berikan analisis yang professional, data-driven, dan actionable."""
        
    def get_system_prompt(self) -> str:
        """System prompt untuk Marketing Agent."""
        return """Anda adalah Marketing Analysis Agent yang ahli dalam analisis pemasaran.

        Berikan analisis yang professional, data-driven, dan actionable."""
    
    def _get_rag_prompt_template(self):
        """Get prompt template untuk RAG."""
        from langchain_core.prompts import PromptTemplate
        
        template = """
        {system_prompt}
        
        Gunakan konteks berikut untuk menjawab pertanyaan:
        {context}
        
        Pertanyaan: {question}
        
        Jika informasi yang dibutuhkan tidak ada dalam konteks di atas, jawab dengan:
        "Informasi tersebut tidak tersedia dalam basis data marketing saya."
        
        Jawaban:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"],
            partial_variables={"system_prompt": self.get_system_prompt()}
        )
    
    def _is_marketing_related(self, query: str) -> bool:
        """Check apakah pertanyaan berkaitan dengan marketing."""
        marketing_keywords = [
            'marketing', 'pemasaran', 'pasar', 'market', 'kompetitor', 'competitor',
            'pelanggan', 'customer', 'konsumen', 'penjualan', 'sales', 'kampanye',
            'campaign', 'brand', 'branding', 'advertising', 'iklan', 'promosi',
            'promotion', 'roi', 'revenue', 'tren', 'trend', 'segmentasi', 'targeting',
            'positioning', 'pricing', 'harga', 'distribusi', 'distribution',
            'digital marketing', 'social media', 'seo', 'sem', 'content marketing',
            'email marketing', 'influencer', 'engagement', 'conversion', 'funnel',
            'analytics', 'metrics', 'kpi', 'ctr', 'cpc', 'cpm', 'roas', 'ltv'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in marketing_keywords)
    
    def generate_response(self, query: str, context: Optional[str] = None, **kwargs) -> str:
        """
        Generate response untuk pertanyaan marketing analysis.
        
        Args:
            query: Pertanyaan user
            context: Konteks tambahan (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Response dari Marketing Agent
        """
        # Check apakah pertanyaan berkaitan dengan marketing
        # if not self._is_marketing_related(query):
        #     return "Maaf, saya hanya dapat membantu dengan pertanyaan seputar analisis marketing dan pemasaran."
        
        try:
            # # Jika ada RetrievalQA dan model OpenAI, gunakan RAG
            # if self.retrieval_qa and self.model_type == "openai":
            #     result = self.retrieval_qa({"query": query})
            #     return result["result"]
            
            # # Fallback ke regular generation dengan vector search
            # else:
                # Search knowledge base
                relevant_docs = self.vector_service.similarity_search(query, k=3)
                
                if relevant_docs:
                    context = "\n".join([doc.page_content for doc in relevant_docs])
                else:
                    context = "Tidak ada informasi relevan dalam knowledge base."
                
                if self.model_type == "openai":
                    client = self._get_model_client()
                    
                    messages = [
                        SystemMessage(content=self.get_system_prompt()),
                        SystemMessage(content=f"Konteks dari knowledge base: {context}"),
                        HumanMessage(content=query)
                    ]
                    
                    response = client.invoke(messages)
                    return response.content
                    
                elif self.model_type == "gemini":
                    client = self._get_model_client()
                    
                    full_prompt = f"{self.get_system_prompt()}\n\n"
                    full_prompt += f"Konteks dari knowledge base: {context}\n\n"
                    full_prompt += f"Pertanyaan: {query}\n\nJawaban:"
                    
                    response = client.generate_content(full_prompt)
                    return response.text
                    
        except Exception as e:
            return f"Maaf, terjadi kesalahan dalam memproses analisis marketing: {str(e)}"
    
    def add_marketing_documents(self, documents: List[str], metadata_list: List[dict] = None):
        """
        Menambahkan dokumen marketing ke knowledge base.
        
        Args:
            documents: List dokumen text
            metadata_list: List metadata untuk setiap dokumen
        """
        try:
            self.vector_service.add_documents(documents, metadata_list)
            # Re-setup RetrievalQA setelah menambah dokumen
            self._setup_retrieval_qa()
            return True
        except Exception as e:
            print(f"Error adding marketing documents: {str(e)}")
            return False
