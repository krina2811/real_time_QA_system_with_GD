import os
import time
from mygdrive import MyGDrive
import PyPDF2
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import pickle
from queue import Queue
from threading import Event
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Set API key for OpenAI
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


class DriveProcessor:
    def __init__(self, credentials_file: str, status_queue: Queue, processing_event: Event):
        self.credentials_file = credentials_file
        self.status_queue = status_queue
        self.processing_event = processing_event
        self.vectorstore_path = "vectorstore"
        self.check_interval = 120  # Check every minute
        self.embeddings = OpenAIEmbeddings()
        self.mydrive = MyGDrive(self.credentials_file)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )


    def get_vectorstore(self) -> FAISS:
        """Get existing vectorstore or create a new one."""
        try:
            if os.path.exists(self.vectorstore_path):
                return FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
            return None
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return None

    def create_or_update_vectorstore(self, texts: List[str], metadatas: List[Dict] = None) -> bool:
        """Create new vectorstore or update existing one."""
        try:
            existing_vectorstore = self.get_vectorstore()
            
            # Create new vectorstore from the current texts
            new_vectorstore = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=metadatas
            )
            
            if existing_vectorstore is not None:
                # If existing vectorstore exists, merge the new one into it
                print("--------------------------------------------------Vectorstore is already available and merge ")
                existing_vectorstore.merge_from(new_vectorstore)
                existing_vectorstore.save_local(self.vectorstore_path)
            else:
                # If no existing vectorstore, save the new one
                print("create new vector store....................................................")
                new_vectorstore.save_local(self.vectorstore_path)
            
            return True
        except Exception as e:
            print(f"Error in create_or_update_vectorstore: {e}")
            return False

    def prepare_document_chunks(self, text: str, metadata: Dict) -> tuple[List[str], List[Dict]]:
        """Prepare document chunks with metadata."""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Prepare metadata for each chunk
            chunk_metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
                chunk_metadatas.append(chunk_metadata)
            
            return chunks, chunk_metadatas
        except Exception as e:
            print(f"Error in prepare_document_chunks: {e}")
            return [], []
        
    def get_pdf_list(self, new_files):
        pdf_list = []
        for file in new_files:
            _, file_name = file["id"], file["name"]
            if file_name.endswith(".pdf"):
                pdf_list.append(file)
        return pdf_list


    def start_processing(self):
        """Main processing loop."""
        while True:
            try:
                # Wait for processing event to be set
                self.processing_event.wait()
                
                # Check for new files
                self.update_status("🔍 Checking for new files...")
                new_files = self.mydrive.get_files()
                pdf_files = self.get_pdf_list(new_files)
                if pdf_files:
                    self.update_status(f"📄 Processing {len(pdf_files)} new files...")
                    
                    for file in pdf_files:
                        try:
                            # Download and process the file
                            file_stream = self.mydrive.download_pdf_to_memory(file['id'])
                            text = self.extract_pdf_text(file_stream)
                            
                            # Prepare metadata
                            metadata = {
                                'file_id': file['id'],
                                'file_name': file['name'],
                                'created_time': file['createdTime'],
                                'modified_time': file['modifiedTime']
                            }
                            
                            # Split text and prepare chunks with metadata
                            chunks, chunk_metadatas = self.prepare_document_chunks(text, metadata)
                            
                            # Update vectorstore with chunked content
                            if chunks and chunk_metadatas:
                                if self.create_or_update_vectorstore(chunks, chunk_metadatas):
                                    self.mydrive.mark_file_as_processed(file['id'])    
                                    self.update_status(f"✅ Processed and indexed:",embeddings_ready=True)
                            else:
                                self.update_status(f"❌ Failed to update index for: {file['name']}")
                            
                        except Exception as e:
                            print(f"Error processing file {file['name']}: {e}")
                            continue
                else:
                    self.update_status("✅ No new files to process", embeddings_ready=True)
                
                # Wait before next check
                time.sleep(self.check_interval)  # Check every minute
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                self.update_status(f"❌ Error in processing: {str(e)}")
                time.sleep(self.check_interval)

    # def process_documents(self, mydrive: MyGDrive):
    #     """Process documents from Google Drive."""
    #     documents = []
    #     print("Getting files from Google Drive...")  # Debug print
    #     new_files = mydrive.get_files()
        
    #     print(f"Found {len(new_files)} new files")  # Debug print
        
    #     if not new_files:
    #         self.update_status("No new files to process", True)
    #         return True

    #     for file in new_files:
    #         try:
    #             # Use direct list access since we know the API returns a list of dicts
    #             file_id = file['id']
    #             file_name = file['name']
                               
    #             if file_name.endswith(".pdf"):
    #                 print(f"Processing file: {file_name}")  # Debug print
    #                 self.update_status(f"Processing file: {file_name}")                
    #                 file_stream = mydrive.download_pdf_to_memory(file_id)
    #                 text = self.extract_pdf_text(file_stream)
                    
    #                 if text:
    #                     documents.append(Document(page_content=text))
    #                     mydrive.mark_file_as_processed(file_id)

    #         except Exception as e:
    #             print(f"Error processing file: {str(e)}")  # Debug print
    #             self.update_status(f"Error processing file {file_name}: {str(e)}")
    #             continue

    #     if documents:
    #         self.update_status("Generating embeddings...")

   
    #         vectorstore = self.get_faiss_vectorstore(documents)
    #         vectorstore.save_local(self.vectorstore_path)  # Save FAISS instead of pickling
                      
    #         self.update_status("Vector store created successfully!", True)
    #         return True

    #     return False

    def extract_pdf_text(self, file_stream):
        """Extract text from a PDF file in memory."""
        reader = PyPDF2.PdfReader(file_stream)
        return "".join(page.extract_text() or "" for page in reader.pages).strip()

    # def get_faiss_vectorstore(self, docs):
    #     """Create FAISS Vectorstore from documents."""
    #     embeddings = OpenAIEmbeddings()
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    #     split_docs = text_splitter.split_documents(docs)
    #     return FAISS.from_documents(split_docs, embeddings)

    def update_status(self, message: str, embeddings_ready: bool = False):
        """Update status through queue."""
        print(f"Status update: {message}")  # Debug print
        self.status_queue.put({
            "message": message,
            "embeddings_ready": embeddings_ready
        })