"""
Aplikasi Streamlit untuk AI Agent dengan dua mode:
1. General Agent - Seperti ChatGPT
2. Marketing Agent - Berbasis RAG untuk analisis marketing
"""
import streamlit as st
from services import AgentService, extract_text_from_pdf, upsert_pdf_to_qdrant
from services.vector_service import VectorService
from utils import setup_logger, validate_model_type, validate_agent_type
from config import settings


# Setup logger
logger = setup_logger("streamlit_app")

# Initialize services
@st.cache_resource
def get_agent_service():
    """Initialize dan cache AgentService."""
    return AgentService()

agent_service = get_agent_service()

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'current_agent' not in st.session_state:
        st.session_state['current_agent'] = 'general'
    if 'current_model' not in st.session_state:
        st.session_state['current_model'] = 'telkom-ai'
    if 'marketing_kb_loaded' not in st.session_state:
        st.session_state['marketing_kb_loaded'] = False

initialize_session_state()

# Page config
st.set_page_config(
    page_title="AI Agent Assistant", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🤖 AI Agent Assistant")
st.markdown("Pilih antara **General Agent** (ChatGPT-like) atau **Marketing Agent** (RAG-based)")

# Sidebar konfigurasi
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # Model Selection
    st.subheader("Model LLM")
    model_options = {
        "telkom-ai": "Telkom AI (v0.0.4)",
        "gemini": "Google Gemini 1.5 Flash"
    }
    
    selected_model = st.selectbox(
        "Pilih Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0 if st.session_state['current_model'] == 'telkom-ai' else 1,
        key="model_selector"
    )
    
    if selected_model != st.session_state['current_model']:
        st.session_state['current_model'] = selected_model
        st.rerun()
    
    # Agent Selection  
    st.subheader("Tipe Agent")
    agent_options = {
        "general": "General Agent",
        "marketing": "Market Analyst Agent"
    }
    
    selected_agent = st.selectbox(
        "Pilih Agent:",
        options=list(agent_options.keys()),
        format_func=lambda x: agent_options[x],
        index=0 if st.session_state['current_agent'] == 'general' else 1,
        key="agent_selector"
    )
    
    if selected_agent != st.session_state['current_agent']:
        st.session_state['current_agent'] = selected_agent
        st.session_state['messages'] = []  # Clear chat history when switching agents
        st.rerun()
    
    # Status
    st.divider()
    st.success(f"**Model Aktif:** {model_options[st.session_state['current_model']]}")
    st.success(f"**Agent Aktif:** {agent_options[st.session_state['current_agent']]}")
    
    # Marketing Agent Knowledge Base
    if st.session_state['current_agent'] == 'marketing':
        st.divider()
        st.subheader("📚 Knowledge Base Marketing")
        
        uploaded_file = st.file_uploader(
            "Upload dokumen marketing (PDF):",
            type=['pdf'],
            help="Upload dokumen berisi informasi marketing untuk analisis yang lebih akurat"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Memproses dokumen..."):
                    # Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    
                    # Add to marketing knowledge base
                    vector_service = VectorService(collection_name=settings.qdrant_marketing_collection)
                    success = vector_service.upsert_documents_from_pdf(
                        text, 
                        metadata={
                            "filename": uploaded_file.name,
                            "type": "marketing_document"
                        }
                    )
                    
                    if success:
                        st.success(f"✅ Dokumen '{uploaded_file.name}' berhasil ditambahkan ke knowledge base!")
                        st.session_state['marketing_kb_loaded'] = True
                        
                        # Preview
                        with st.expander("🔍 Preview Dokumen"):
                            st.text_area("Isi dokumen:", value=text[:1000] + "..." if len(text) > 1000 else text, height=200, disabled=True)
                    else:
                        st.error("❌ Gagal menambahkan dokumen ke knowledge base")
                        
            except Exception as e:
                st.error(f"❌ Error memproses dokumen: {str(e)}")
        
        # Knowledge base status
        if st.session_state['marketing_kb_loaded']:
            st.info("📋 Knowledge base marketing siap digunakan")
        else:
            st.warning("Upload dokumen marketing untuk hasil analisis yang lebih akurat")

# Agent Info
col1, col2 = st.columns([2, 1])

with col1:
    # Agent description
    if st.session_state['current_agent'] == 'general':
        st.info("""
        **General Agent** - Assistant AI seperti ChatGPT yang dapat membantu dengan:
        - Pertanyaan umum tentang teknologi, sains, bisnis
        - Menulis dan editing
        - Analisis dan perhitungan
        - Brainstorming dan diskusi
        - Dan berbagai topik lainnya
        """)
    else:
        st.info("""
        **Marketing Analyst Agent** - Specialist analisis marketing berbasis RAG yang fokus pada:
        - Analisis tren pasar dan industri
        - Analisis kompetitor dan positioning
        - Segmentasi dan targeting pelanggan
        - Strategi penjualan dan distribution
        - Evaluasi kampanye marketing
        - ROI dan marketing metrics
        
        ⚠️ **Hanya menjawab pertanyaan seputar marketing!**
        """)

with col2:
    # Clear chat button
    if st.button("🗑️ Hapus Riwayat Chat", type="secondary"):
        st.session_state['messages'] = []
        st.rerun()

# Chat interface
st.divider()

# Display chat history
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Tanyakan sesuatu..."):
    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Sedang berpikir..."):
            try:
                response = agent_service.chat(
                    query=prompt,
                    agent_type=st.session_state['current_agent'],
                    model_type=st.session_state['current_model']
                )
                
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state['messages'].append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Maaf, terjadi kesalahan: {str(e)}"
                st.error(error_msg)
                st.session_state['messages'].append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 50px;'>
    <small>
        Powered by Telkom AI (v0.0.4) & Google Gemini 1.5 Flash | 
        Vector Store: Qdrant | 
        Framework: LangChain
    </small>
</div>
""", unsafe_allow_html=True)
