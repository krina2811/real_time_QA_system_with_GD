import streamlit as st
import pickle
import os
import json
import time
from queue import Queue
from threading import Thread, Event
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from google_drive_processing import DriveProcessor
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import logging
from typing import List, Dict, Any
from langchain.schema import Document
from pydantic import Field
import hashlib
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# VECTORSTORE_PATH = "vectorstore.pkl"
TEMP_CREDENTIALS_FILE = "temp_credentials.json"

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "embeddings_ready" not in st.session_state:
        st.session_state.embeddings_ready = False
    if "status" not in st.session_state:
        st.session_state.status = "⏳ Please upload your Google Cloud Service Key to begin."
    if "processor" not in st.session_state:
        st.session_state.processor = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "processing_thread" not in st.session_state:
        st.session_state.processing_thread = None
    if "last_vectorstore_update" not in st.session_state:
        st.session_state.last_vectorstore_update = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = {}
    if "last_vectorstore_hash" not in st.session_state:
        st.session_state.last_vectorstore_hash = None  # Initialize with None

def get_directory_hash(directory):
    """Generate a SHA-256 hash based on the content of all files in the directory."""
    hasher = hashlib.sha256()
    faiss_files = glob.glob(os.path.join(directory, "*"))  # Get all FAISS files

    if not faiss_files:
        return None  # No FAISS files found

    for file in sorted(faiss_files):  # Sort files for consistent hashing
        try:
            with open(file, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}", exc_info=True)
            return None

    return hasher.hexdigest()  # Return final hash



class DebugRetriever:
    """Wrapper class for retriever to add debugging capabilities"""
    vectorstore: FAISS = Field(description="FAISS vectorstore")
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "k": 5,
            "score_threshold": 0.2,
            "fetch_k": 20
        }
    )

    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, vectorstore: FAISS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.vectorstore = vectorstore
        self.search_kwargs = {
            "k": 5,
            "score_threshold": 0.2,
            "fetch_k": 20
        }
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents with debug information"""
        # Perform similarity search with scores
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, 
            k=self.search_kwargs["k"],
            fetch_k=self.search_kwargs["fetch_k"]
        )
        
        # Store debug information
        st.session_state.debug_info['retrieved_docs'] = [
            {
                'content': doc.page_content[:200] + "...",
                'metadata': doc.metadata,
                'similarity': score
            }
            for doc, score in docs_and_scores
        ]
        
        # Filter by score threshold if specified
        if self.search_kwargs.get("score_threshold"):
            docs_and_scores = [
                (doc, score) for doc, score in docs_and_scores 
                if score >= self.search_kwargs["score_threshold"]
            ]
        
        # Return only the documents
        return [doc for doc, _ in docs_and_scores]
    
def get_qa_chain(vectorstore):
    """Initialize the QA chain with the vectorstore."""

    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template="""
        Given the following conversation and a question, help provide an informative answer based on the context. 
        If you can't find the answer in the context, say so clearly.

        Chat History:
        {chat_history}

        Question:
        {question}

        Context:
        {context}

        Please provide a clear and concise answer based on the context above. 
        If information comes from multiple documents, try to synthesize it coherently.

        Answer:"""
    )
    # retriever = vectorstore.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={
    #         "k": 5,  
    #         "score_threshold": 0.3  
    #     }
    # )

    retriever = DebugRetriever(vectorstore)
    
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer", 
        return_messages=True
    )
    
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0.7,
        max_tokens=2000
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        # return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        verbose=True
    )
    
    return qa_chain

def create_temp_credentials_file(service_account_info):
    """Create a temporary file with service account credentials."""
    if os.path.exists(TEMP_CREDENTIALS_FILE):
        os.remove(TEMP_CREDENTIALS_FILE)

    try:
        with open(TEMP_CREDENTIALS_FILE, 'w') as f:
            json.dump(service_account_info, f)
        return True
    except Exception as e:
        st.error(f"Error creating credentials file: {e}")
        return False


# def check_vectorstore_updates():
#     """Check if vectorstore needs to be reloaded and update QA chain."""
#     try:
#         if os.path.exists("vectorstore"):
#             # last_modified = os.path.getmtime("vectorstore")
#             last_modified = os.stat("vectorstore").st_mtime
#             current_size = os.path.getsize("vectorstore")
            
#             logger.info(f"Checking vectorstore - Size: {current_size}, Last Modified: {last_modified}")
            
#             if (st.session_state.last_vectorstore_update is None or last_modified > st.session_state.last_vectorstore_update):
                
#                 # Load new vectorstore
#                 vectorstore = FAISS.load_local(
#                     "vectorstore", 
#                     OpenAIEmbeddings(), 
#                     allow_dangerous_deserialization=True
#                 )
                
#                 # Get index stats
#                 index_stats = {
#                     'num_vectors': vectorstore.index.ntotal,
#                     'dimension': vectorstore.index.d,
#                 }
                
#                 logger.info(f"Loaded vectorstore stats: {index_stats}")
#                 st.session_state.debug_info['vectorstore_stats'] = index_stats
                
#                 # Update session state
#                 st.session_state.vectorstore = vectorstore
#                 st.session_state.qa_chain = get_qa_chain(vectorstore)
#                 st.session_state.last_vectorstore_update = last_modified
#                 st.session_state.embeddings_ready = True
                
#                 # Clear memory when vectorstore is updated
#                 if st.session_state.qa_chain and hasattr(st.session_state.qa_chain, 'memory'):
#                     st.session_state.qa_chain.memory.clear()
                
#                 return True
#     except Exception as e:
#         logger.error(f"Error in vectorstore update: {str(e)}", exc_info=True)
#         st.error(f"Error checking vectorstore updates: {e}")
#     return False


def check_vectorstore_updates():
    """Check if vectorstore needs to be reloaded and update QA chain."""
    try:
        directory_hash = get_directory_hash("vectorstore")  # Get hash of FAISS files
        if directory_hash is None:
            logger.warning("No FAISS index files found in 'vectorstore'.")
            return False

        # Compare with previous hash stored in session state
        if (st.session_state.last_vectorstore_hash != directory_hash):

            logger.info(f"Reloading vectorstore due to updates... (Hash: {directory_hash})")

            # Load new vector store
            vectorstore = FAISS.load_local(
                "vectorstore", 
                OpenAIEmbeddings(), 
                allow_dangerous_deserialization=True
            )

            # Get index stats
            index_stats = {
                'num_vectors': vectorstore.index.ntotal,
                'dimension': vectorstore.index.d,
            }

            logger.info(f"Updated vectorstore stats: {index_stats}")
            st.session_state.debug_info['vectorstore_stats'] = index_stats

            # Update session state
            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = get_qa_chain(vectorstore)
            st.session_state.last_vectorstore_hash = directory_hash  # Store new hash
            st.session_state.embeddings_ready = True

            # Clear memory after update
            if st.session_state.qa_chain and hasattr(st.session_state.qa_chain, 'memory'):
                st.session_state.qa_chain.memory.clear()

            return True

    except Exception as e:
        logger.error(f"Error in vectorstore update: {str(e)}", exc_info=True)
        st.error(f"Error checking vectorstore updates: {e}")

    return False


def main():
    """Main Streamlit UI for chatbot."""
    st.title("Chat with Google Drive Documents")
    
    # Initialize session state
    init_session_state()

    
    # Status display
    status_container = st.empty()
    status_container.info(st.session_state.status)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        gcloud_key = st.file_uploader(
            "Upload Google Cloud Service Key (JSON)",
            type=["json"],
            help="Upload your service account key to process documents"
        )
        
        if gcloud_key and st.session_state.processor is None:
            try:
                service_account_info = json.load(gcloud_key)
                if create_temp_credentials_file(service_account_info):
                    # st.success("✅ Credentials uploaded successfully!")
                    st.session_state.status = "✅ Credentials uploaded successfully!"
                    status_container.info(st.session_state.status)
                    
                    # Initialize the processor with a queue for status updates
                    status_queue = Queue()
                    processing_event = Event()
                    processing_event.set()  # Start in active state
                    
                    processor = DriveProcessor(
                        TEMP_CREDENTIALS_FILE,
                        status_queue,
                        processing_event
                    )
                    st.session_state.processor = processor
                    
                    st.session_state.status = "✅ Document Processing started !!"
                    status_container.info(st.session_state.status)
                    # Start processing in a separate thread
                    if st.session_state.processing_thread is None:
                        thread = Thread(
                            target=processor.start_processing,
                            daemon=True
                        )
                        st.session_state.processing_thread = thread
                        thread.start()
                        
            except Exception as e:
                st.error(f"Error processing service account key: {e}")
    
    # Update status from processor if available
    if st.session_state.processor:
        status_queue = st.session_state.processor.status_queue
        while not status_queue.empty():
            status = status_queue.get()
            st.session_state.status = status.get("message", "")
            status_container.info(st.session_state.status)
            
            # Check for vectorstore updates if new documents were processed
            if "Processed and indexed" in st.session_state.status:
                check_vectorstore_updates()
        
                
    
    if st.session_state.embeddings_ready and st.session_state.vectorstore:
        # Display chat history
        for message in st.session_state.chat_history:
            role = message.get("role", "")
            content = message.get("content", "")
            with st.chat_message(role):
                st.write(content)
        
        # User input
        user_query = st.text_input("Ask a question about your documents:")
        
        if st.button("Ask") and user_query:
            # Pause document processing during query
            if st.session_state.processor:
                st.session_state.processor.processing_event.clear()
            
            try:
                # Always check for updates before processing query
                check_vectorstore_updates()
                
                with st.spinner("Generating answer..."):
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_query
                    })
                    
                    # Get response from QA chain
                    response = st.session_state.qa_chain({
                        "question": user_query
                    })
                    
                    answer = response.get("answer", "I couldn't find a relevant answer.")
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    # Display answer
                    with st.chat_message("assistant"):
                        st.write(answer)
                    
                    # Display source documents if available
                    if "source_documents" in response:
                        with st.expander("View Source Documents"):
                            for i, doc in enumerate(response["source_documents"], 1):
                                st.markdown(f"""
                                **Source {i}**
                                - File: {doc.metadata.get('file_name', 'Unknown')}
                                - Content: {doc.page_content}
                                """)
                                            
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                print(f"Detailed error in QA process: {str(e)}")
            
            finally:
                # Resume document processing
                if st.session_state.processor:
                    st.session_state.processor.processing_event.set()
    
    # Add a small delay to prevent excessive CPU usage
    time.sleep(0.1)
    st.rerun()
if __name__ == "__main__":
    main()