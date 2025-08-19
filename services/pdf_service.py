"""
PDF Service untuk ekstraksi dan processing PDF.
"""
import PyPDF2
from io import BytesIO
from typing import List, Dict, Any
from services.vector_service import VectorService
from config import settings


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text dari uploaded PDF file.
    
    Args:
        uploaded_file: Uploaded file dari Streamlit
        
    Returns:
        Extracted text dari PDF
    """
    try:
        # Read file sebagai bytes
        pdf_bytes = uploaded_file.read()
        pdf_file = BytesIO(pdf_bytes)
        
        # Extract text menggunakan PyPDF2
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")


def upsert_pdf_to_qdrant(text: str, metadata: Dict[str, Any], collection_name: str = None) -> bool:
    """
    Upsert PDF text ke Qdrant vector store.
    
    Args:
        text: Text dari PDF
        metadata: Metadata dokumen
        collection_name: Nama collection (default: dari settings)
        
    Returns:
        True jika berhasil
    """
    try:
        collection_name = collection_name or settings.qdrant_collection_name
        vector_service = VectorService(collection_name=collection_name)
        return vector_service.upsert_documents_from_pdf(text, metadata)
    
    except Exception as e:
        print(f"Error upserting PDF to Qdrant: {str(e)}")
        return False


def search_knowledge_base(query: str, top_k: int = 3, collection_name: str = None) -> List[str]:
    """
    Search knowledge base untuk informasi relevan.
    
    Args:
        query: Query untuk search
        top_k: Jumlah hasil yang dikembalikan
        collection_name: Nama collection
        
    Returns:
        List text hasil search
    """
    try:
        collection_name = collection_name or settings.qdrant_collection_name
        vector_service = VectorService(collection_name=collection_name)
        docs = vector_service.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]
    
    except Exception as e:
        print(f"Error searching knowledge base: {str(e)}")
        return []
