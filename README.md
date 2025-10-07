# 🎥 Chat with YouTube Videos 🤖

A **Streamlit-based AI assistant** that lets users ask questions about any YouTube video using its transcript.  
The app fetches video captions via **YouTube Transcript API**, splits long transcripts into manageable chunks, generates embeddings using **Google Generative AI**, and builds a **FAISS retriever** for semantic search.

Users can type questions, and the AI provides answers **based solely on the video transcript**, ensuring context-aware responses. The system gracefully handles errors like unavailable transcripts or embedding failures.

**Key Features:**
- Automatic transcript fetching and text processing  
- Semantic search and context-aware question answering  
- Uses advanced NLP and embeddings for accurate answers  
- Modular design allows easy extension to other video platforms or LLMs  

**Tech Stack:** Python, Streamlit, LangChain, Google Generative AI, FAISS, YouTube Transcript API  

**Usage:**  
1. Enter the YouTube video ID (not the full URL).  
2. Click "Load Transcript" to fetch and index it.  
3. Type your question and get AI-generated answers based on the transcript.  

