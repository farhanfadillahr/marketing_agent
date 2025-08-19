"""
Vector Service untuk mengelola Qdrant vector database.
"""
from typing import List, Optional, Dict, Any
# from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import settings
import uuid


class VectorService:
    """Service untuk mengelola operasi vector database dengan Qdrant."""
    
    def __init__(self, collection_name: str = None):
        """
        Initialize VectorService.
        
        Args:
            collection_name: Nama collection Qdrant
        """
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.client = self._get_qdrant_client()
        self.embeddings = self._get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorstore = None
        self._ensure_collection_exists()
    
    def _get_qdrant_client(self) -> QdrantClient:
        """Get Qdrant client."""
        return QdrantClient(
            url=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=settings.qdrant_is_https
        )
    
    def _get_embeddings(self):
        """Get embeddings model."""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key
        )
        return self.embeddings
    
    def _ensure_collection_exists(self):
        """Pastikan collection exists di Qdrant."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # Size untuk text-embedding-3-small
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            
            # Initialize vectorstore
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
            
        except Exception as e:
            print(f"Error ensuring collection exists: {str(e)}")
    
    def add_documents(self, documents: List[str], metadata_list: List[Dict] = None) -> bool:
        """
        Menambahkan dokumen ke vector store.
        
        Args:
            documents: List dokumen text
            metadata_list: List metadata untuk setiap dokumen
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            # Split documents
            all_chunks = []
            for i, doc_text in enumerate(documents):
                chunks = self.text_splitter.split_text(doc_text)
                for j, chunk in enumerate(chunks):
                    metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                    metadata.update({
                        'chunk_id': f"{i}_{j}",
                        'doc_id': str(i),
                        'chunk_index': j
                    })
                    all_chunks.append(Document(page_content=chunk, metadata=metadata))
            
            # Add to vectorstore
            if self.vectorstore and all_chunks:
                self.vectorstore.add_documents(all_chunks)
                return True
            return False
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Melakukan similarity search.
        
        Args:
            query: Query untuk search
            k: Jumlah dokumen yang dikembalikan
            
        Returns:
            List dokumen yang relevan
        """
        try:
            if self.vectorstore:
                return self.vectorstore.similarity_search(query, k=k)
            return []
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """
        Melakukan similarity search dengan score.
        
        Args:
            query: Query untuk search
            k: Jumlah dokumen yang dikembalikan
            
        Returns:
            List tuple (document, score)
        """
        try:
            if self.vectorstore:
                return self.vectorstore.similarity_search_with_score(query, k=k)
            return []
        except Exception as e:
            print(f"Error in similarity search with score: {str(e)}")
            return []
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Dict = None):
        """
        Get retriever untuk RAG.
        
        Args:
            search_type: Tipe search ('similarity', 'mmr', dll)
            search_kwargs: Kwargs untuk search
            
        Returns:
            Retriever object
        """
        try:
            if self.vectorstore:
                search_kwargs = search_kwargs or {"k": 3}
                return self.vectorstore.as_retriever(
                    search_type=search_type,
                    search_kwargs=search_kwargs
                )
            return None
        except Exception as e:
            print(f"Error getting retriever: {str(e)}")
            return None
    
    def delete_collection(self) -> bool:
        """
        Hapus collection.
        
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            self.client.delete_collection(self.collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get informasi collection.
        
        Returns:
            Dict informasi collection
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size if info.config else 0,
                "vectors_count": info.points_count if hasattr(info, 'points_count') else 0,
                "status": info.status if hasattr(info, 'status') else 'unknown'
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {}
    
    def upsert_documents_from_pdf(self, text: str, metadata: Dict = None) -> bool:
        """
        Upsert dokumen dari PDF text.
        
        Args:
            text: Text dari PDF
            metadata: Metadata dokumen
            
        Returns:
            True jika berhasil
        """
        try:
            metadata = metadata or {}
            return self.add_documents([text], [metadata])
        except Exception as e:
            print(f"Error upserting PDF documents: {str(e)}")
            return False
