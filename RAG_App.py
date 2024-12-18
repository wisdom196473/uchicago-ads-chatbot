import streamlit as st
from streamlit_chat import message
import torch
import re
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add this line after loading environment variables
Settings.embed_model = OpenAIEmbedding()

# Set page title and favicon
st.set_page_config(page_title="UChicago ADS Chatbot", page_icon="🎓")

# Load the fine-tuned model from Hugging Face
@st.cache_resource
def load_model():
    return SentenceTransformer("chen196473/UChicago-ADS-Bge-Large", device="cuda" if torch.cuda.is_available() else "cpu")

fine_tuned_model = load_model()

# Define embedding function using fine-tuned model
def get_fine_tuned_embedding(text):
    return fine_tuned_model.encode(text, convert_to_tensor=True).cpu().numpy()

# Set API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define query prompt template
query_prompt_template = PromptTemplate(
    template=(
        "You are an AI assistant for the University of Chicago's Applied Data Science program. "
        "Your role is to provide accurate and concise information exactly as it appears in the source material. "
        "Follow these rules:\n"
        "1. Provide answers in the exact format and wording as shown in the source material\n"
        "2. When the source material contains URLs or webpage references, include them in markdown format [text](url)\n"
        "3. Keep responses concise and direct\n"
        "4. Maintain the exact formatting, including bullet points and line breaks\n"
        "5. If multiple sources contain the same information, use the most recent or most complete version\n\n"
        "6. For tuition questions, provide the answer in this exact format, including dollar signs and no spaces around slashes:\n"
        "   'Tuition for the MS in Applied Data Science program: $X per course / $Y total tuition'\n"     
        "Context: {context_str}\n\n"
        "Question: {query_str}\n\n"
        "Answer: "
    )
)

# Load or build index
@st.cache_resource
def build_or_load_index(index_path="./saved_index"):
    if os.path.exists(index_path):
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
    else:
        st.error("Index not found. Please build the index first.")
        return None

    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=query_prompt_template
    )

    return index.as_query_engine(
        similarity_top_k=10,
        response_synthesizer=response_synthesizer,
        structured_answer_filtering=True
    )

llama_query_engine = build_or_load_index()

# Streamlit app
st.title("UChicago ADS Chat Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
def format_tuition_response(response):
    # Regular expression to match and format the tuition information
    pattern = r"Tuition for the MS in Applied Data Science program: \$?([\d,]+) per course/\$?([\d,]+) total tuition"
    match = re.search(pattern, response)
    if match:
        per_course, total = match.groups()
        return f"Tuition for the MS in Applied Data Science program: ${per_course} per course/${total} total tuition"
    return response  # Return original response if no match found

# React to user input
if prompt := st.chat_input("Ask me about UChicago's Applied Data Science program"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if llama_query_engine:
        response = llama_query_engine.query(prompt)
        formatted_response = format_tuition_response(response.response)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(formatted_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
    else:
        st.error("Query engine not initialized. Please check the index.")

# Add a sidebar with information about the chatbot
st.sidebar.title("About")
st.sidebar.info(
    "This chatbot provides information about the University of Chicago's "
    "Applied Data Science program."
)

# Add author names to the sidebar
st.sidebar.title("Authors")
st.sidebar.markdown("Yu-Chih (Wisdom) Chen")

# Add Reset Chat button to the sidebar
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.rerun()

# Add footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
    }
    </style>
    <div class="footer">
        <p>Powered by UChicago ADS Program | © 2024 University of Chicago</p>
    </div>
    """,
    unsafe_allow_html=True
)
