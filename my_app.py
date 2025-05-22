import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

st.title("Chat with YouTube Videos 🤖")

# Input YouTube video ID
video_id = st.text_input("Enter the YouTube video ID (not the full URL) :")

if st.button('Load Transcript'):
    with st.spinner("⏳Fetching and processing transcript..."):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)  
            transcript = " ".join(chunk['text'] for chunk in transcript_list)
            
            # Text splitting
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])
            
            # Generate embeddings
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
                vector_store = FAISS.from_documents(chunks,embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 4})
                
                st.session_state.retriever = retriever
                st.success("✅ Transcript indexed and retriever ready.")
            except Exception as embed_error:
                st.error(f"Embedding error: {embed_error}")
        except TranscriptsDisabled:
            st.error("No captions available for this video.")
        except Exception as e:
            st.error(f"Failed to load transcript: {e}")
            
if "retriever" in st.session_state:
    st.subheader("Ask a question: ")
    user_qs = st.text_input("Your question: ")
    
    if st.button("Get answer"):
        with st.spinner("Generating answer..."):
            try:
                # Prompt template
                prompt = PromptTemplate(
                    template = """ 
                        You are a helpful assistant.
                        Answer ONLY from the provided transcript context.
                        If the context is insufficient, just say you don't know.
                        
                        {context}
                        Question : {question}
                        """,
                    input_variables = ['context','question']
                                       )
                # Format context
                def format_docs(retrieved_docs):
                    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    return context_text
                
                # Define LLM
                llm = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')
                
                # Creating Parser
                parser = StrOutputParser()
                
                # Forming Chain
                retriever = st.session_state.retriever
                parallel_chain = RunnableParallel({
                    'context' : retriever | RunnableLambda(format_docs),
                    'question' : RunnablePassthrough()
                })
                
                main_chain = parallel_chain | prompt | llm | parser
                
                # Invoke Chain
                answer = main_chain.invoke(user_qs)
                st.success("✅ Answer generated:")
                st.write(answer)
                
            except Exception as generation_error:
                st.error(f"Generation error: {generation_error}")