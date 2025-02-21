# Real-Time Question-Answering System with Dynamic Vector Store Updates

## 📌 Overview
This project implements a **real-time question-answering (QA) system** that continuously monitors **Google Drive** for new document uploads, processes them, and updates the **vector database** dynamically. The system is designed using **Retrieval-Augmented Generation (RAG)**, ensuring that newly uploaded documents are indexed and ready for question answering without recomputing all embeddings.

## 🚀 Features
- **Real-time Google Drive monitoring** for new document uploads.  
- **Automatic embedding generation** for new documents.  
- **Efficient vector database update** (only adding new embeddings instead of recomputing all).  
- **Multiprocessing architecture** for parallel document tracking and QA processing.  
- **RAG-based question answering** with conversation history.  
- **Scalable and optimized for fast retrieval**.

## 🛠️ Tech Stack
- **Python**  
- **LangChain**  
- **OpenAI API / LLMs**  
- **Vector Database (e.g., FAISS)**  
- **Google Drive API**  
- **Streamlit** (for UI)  
- **MultiThreding** (for parallel execution)  

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
git clone https://github.com/krina2811/real_time_QA_system_with_GD.git 

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Set Up Google Drive API
To create the credentials.json file, please follow the detailed steps provided in the following guide:
[Google Service Account Setup Guide](https://github.com/colinmcnamara/austin_langchain/blob/main/labs/LangChain_103/rag_ollama_llava_drive/GoogleServiceAccount.md)

### 4️⃣ Run the System
streamlit run chat_interface.py

## ⚙️ System Workflow

1. **Google Drive Monitoring**  
   - Continuously tracks Google Drive for new document uploads.  
   - Detects changes and identifies newly added files.  

2. **Embeddings Generation**  
   - Extracts text from the uploaded documents.  
   - Converts the text into vector embeddings using a pre-trained model.  

3. **Vector Store Update**  
   - Adds only new embeddings to the vector database.  
   - Avoids recomputing embeddings for previously indexed documents.  

4. **Question-Answering Module**  
   - Uses **retrieval-augmented generation (RAG)** to fetch relevant document snippets.  
   - Provides accurate answers based on both historical and newly uploaded documents.  

5. **User Query Processing**  
   - Accepts user questions through an interface (Streamlit UI).  
   - Retrieves relevant document sections and generates responses.  

6. **Parallel Processing for Efficiency**  
   - **One process:** Monitors and updates embeddings in real time.  
   - **Another process:** Handles user queries using the latest indexed data.  

## 📈 Future Enhancements
- Support for more document formats (e.g., PDFs, Word, CSV).  
- Integrate advanced LLMs for better response generation.  
- Implement authentication for secured access.  
- Implement advanced RAG techniques for better response accuracy and relevance.  
