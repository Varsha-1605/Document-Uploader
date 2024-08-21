# # # __import__('pysqlite3')
# # # import sys
# # # sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# # # import tempfile
# # # import os
# # # import time
# # # import streamlit as st
# # # from dotenv import load_dotenv
# # # from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# # # from langchain_community.document_loaders import (
# # #     PyPDFLoader,
# # #     UnstructuredFileLoader,
# # #     UnstructuredWordDocumentLoader
# # # )
# # # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # # from langchain_chroma import Chroma
# # # from langchain.agents.agent_toolkits import (
# # #     create_vectorstore_agent,
# # #     VectorStoreToolkit,
# # #     VectorStoreInfo
# # # )
# # # from langchain.memory import ConversationBufferMemory
# # # from langchain.callbacks.streamlit import StreamlitCallbackHandler
# # # import json
# # # import pandas as pd
# # # import plotly.express as px
# # # import tenacity
# # # import httpx
# # # import asyncio

# # # # Load environment variables from .env file
# # # load_dotenv()

# # # # Set page configuration
# # # st.set_page_config(page_title="OmniQuery", page_icon="🧠", layout="wide")

# # # # Custom CSS for dark theme and improved UI
# # # st.markdown("""
# # # <style>
# # #     @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
# # #     body {
# # #         font-family: 'Roboto', sans-serif;
# # #         background-color: #1e1e1e;
# # #         color: #f0f0f0;
# # #     }
# # #     .stApp {
# # #         max-width: 1200px;
# # #         margin: 0 auto;
# # #     }
# # #     .main {
# # #         padding: 2rem;
# # #         border-radius: 10px;
# # #         background-color: #2d2d2d;
# # #         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
# # #     }
# # #     h1, h2, h3 {
# # #         color: #bb86fc;
# # #     }
# # #     .stButton>button {
# # #         background-color: #03dac6;
# # #         color: #000000;
# # #         border: none;
# # #         padding: 0.5rem 1rem;
# # #         border-radius: 5px;
# # #         font-weight: bold;
# # #         transition: all 0.3s ease;
# # #     }
# # #     .stButton>button:hover {
# # #         background-color: #018786;
# # #         color: #ffffff;
# # #         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
# # #     }
# # #     .stTextInput>div>div>input {
# # #         background-color: #3d3d3d;
# # #         color: #f0f0f0;
# # #         border: 2px solid #bb86fc;
# # #         border-radius: 5px;
# # #         padding: 0.5rem;
# # #     }
# # #     .chat-message {
# # #         padding: 1.5rem;
# # #         border-radius: 0.5rem;
# # #         margin-bottom: 1rem;
# # #         display: flex;
# # #         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
# # #     }
# # #     .chat-message.user {
# # #         background-color: #4a4a4a;
# # #     }
# # #     .chat-message.bot {
# # #         background-color: #3d3d3d;
# # #     }
# # #     .chat-message .avatar {
# # #         width: 15%;
# # #     }
# # #     .chat-message .message {
# # #         width: 85%;
# # #         padding: 0 1.5rem;
# # #     }
# # #     .feedback-button {
# # #         padding: 0.3rem 0.8rem;
# # #         border-radius: 15px;
# # #         border: none;
# # #         color: #000000;
# # #         cursor: pointer;
# # #         margin-right: 0.5rem;
# # #         font-size: 0.8rem;
# # #     }
# # #     .feedback-button.positive {
# # #         background-color: #03dac6;
# # #     }
# # #     .feedback-button.negative {
# # #         background-color: #cf6679;
# # #     }
# # #     .feedback-button:hover {
# # #         opacity: 0.8;
# # #     }
# # #     .stProgress > div > div > div > div {
# # #         background-color: #bb86fc;
# # #     }
# # #     .stSelectbox > div > div > div {
# # #         background-color: #3d3d3d;
# # #         color: #f0f0f0;
# # #     }
# # #     .stSelectbox > div > div > div > div {
# # #         background-color: #2d2d2d;
# # #     }
# # #     .stExpander {
# # #         background-color: #3d3d3d;
# # #         border-radius: 5px;
# # #     }
# # # </style>
# # # """, unsafe_allow_html=True)

# # # # Header
# # # st.markdown('<h1 style="text-align: center;">🧠 OmniQuery: Document Intelligence Platform</h1>', unsafe_allow_html=True)
# # # st.markdown('<p style="text-align: center;">🚀 Upload your documents and ask any questions. OmniQuery extracts and analyzes information for you!</p>', unsafe_allow_html=True)

# # # # Set up OpenAI API key from .env file
# # # openai_api_key = os.getenv("OPENAI_API_KEY")
# # # if not openai_api_key:
# # #     st.error("❌ OpenAI API key not found. Please check your .env file.")
# # #     st.stop()

# # # # Create a custom HTTP client with retries and timeouts
# # # @st.cache_resource
# # # def get_http_client():
# # #     return httpx.AsyncClient(
# # #         timeout=httpx.Timeout(60.0, connect=30.0),
# # #         limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
# # #         transport=httpx.AsyncHTTPTransport(retries=3)
# # #     )

# # # # Initialize OpenAI Chat model with streaming and retry logic
# # # @tenacity.retry(
# # #     wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
# # #     stop=tenacity.stop_after_attempt(5),
# # #     reraise=True
# # # )
# # # def get_llm():
# # #     return ChatOpenAI(
# # #         temperature=0.7, 
# # #         model_name="gpt-3.5-turbo",
# # #         streaming=True,
# # #         openai_api_key=openai_api_key
# # #     )

# # # # Initialize embeddings with retry logic
# # # @tenacity.retry(
# # #     wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
# # #     stop=tenacity.stop_after_attempt(5),
# # #     reraise=True
# # # )
# # # def get_embeddings():
# # #     return OpenAIEmbeddings(openai_api_key=openai_api_key)

# # # # Function to load document based on file type
# # # def load_document(file_path):
# # #     file_extension = os.path.splitext(file_path)[1].lower()
# # #     if file_extension == '.pdf':
# # #         return PyPDFLoader(file_path)
# # #     elif file_extension == '.docx':
# # #         return UnstructuredWordDocumentLoader(file_path)
# # #     elif file_extension in ['.txt', '.md']:
# # #         return UnstructuredFileLoader(file_path)
# # #     else:
# # #         raise ValueError(f"Unsupported file type: {file_extension}")

# # # # Process document function with chunking and progress bar
# # # def process_document(file_path):
# # #     loader = load_document(file_path)
# # #     documents = loader.load()
    
# # #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# # #     texts = text_splitter.split_documents(documents)
    
# # #     embeddings = get_embeddings()
    
# # #     progress_bar = st.progress(0)
# # #     store = None
# # #     chunk_size = 100  # Process 100 texts at a time
    
# # #     for i in range(0, len(texts), chunk_size):
# # #         chunk = texts[i:i+chunk_size]
# # #         if store is None:
# # #             store = Chroma.from_documents(chunk, embeddings, collection_name='document_store')
# # #         else:
# # #             store.add_documents(chunk)
# # #         progress = min((i + chunk_size) / len(texts), 1.0)
# # #         progress_bar.progress(progress)
    
# # #     progress_bar.empty()
# # #     return store

# # # # Initialize session state
# # # if 'chat_history' not in st.session_state:
# # #     st.session_state.chat_history = []
# # # if 'feedback' not in st.session_state:
# # #     st.session_state.feedback = {}
# # # if 'document_processed' not in st.session_state:
# # #     st.session_state.document_processed = False

# # # # Sidebar for file upload and options
# # # with st.sidebar:
# # #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px">', unsafe_allow_html=True)
# # #     st.header("Document Upload 📂")
# # #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px">', unsafe_allow_html=True)
# # #     uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt', 'md'])
    

# # #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px">', unsafe_allow_html=True)
# # #     st.header("Options ⚙️")
# # #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px">', unsafe_allow_html=True)
# # #     temperature = st.slider("🌡️ AI Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)



# # #     # Feedback Section
# # #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px;">', unsafe_allow_html=True)
# # #     st.header('Feedback 📝')
# # #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px;">', unsafe_allow_html=True)
# # #     st.sidebar.markdown('<p>We value your feedback! 😊</p>', unsafe_allow_html=True)

# # #     feedback = st.sidebar.slider("How helpful is this tool? 😞😐😊", 1, 5, 3)
# # #     feedback_text = st.sidebar.text_area("Additional feedback:")
# # #     feedback_button = st.sidebar.button("Submit Feedback")
# # #     if feedback_button:
# # #         st.sidebar.success("Thank you for your feedback! 👍")
    
# # #     if uploaded_file:
# # #         st.success(f"✅ Uploaded: {uploaded_file.name}")

# # # # Main content area
# # # if uploaded_file is not None and not st.session_state.document_processed:
# # #     # Save uploaded file temporarily
# # #     with st.spinner("🔄 Processing document..."):
# # #         temp_dir = tempfile.mkdtemp()
# # #         if not os.path.exists(temp_dir):
# # #             os.makedirs(temp_dir)
# # #         file_path = os.path.join(temp_dir, uploaded_file.name)
# # #         with open(file_path, "wb") as f:
# # #             f.write(uploaded_file.getbuffer())

# # #         try:
# # #             store = process_document(file_path)

# # #             # Create vectorstore info object
# # #             vectorstore_info = VectorStoreInfo(
# # #                 name='document_store',
# # #                 description='Processed document for Q&A',
# # #                 vectorstore=store
# # #             )

# # #             # Add the memory to the agent
# # #             memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # #             # Create the LLM
# # #             llm = get_llm()

# # #             # Convert the document store into a langchain toolkit
# # #             toolkit = VectorStoreToolkit(
# # #                 vectorstore_info=vectorstore_info,
# # #                 llm=llm
# # #             )

# # #             # Add the toolkit to an end-to-end LC
# # #             agent_executor = create_vectorstore_agent(
# # #                 llm=llm,
# # #                 toolkit=toolkit,
# # #                 verbose=True,
# # #                 agent_kwargs={"memory": memory}
# # #             )

# # #             st.session_state['agent_executor'] = agent_executor
# # #             st.session_state['memory'] = memory
# # #             st.session_state.document_processed = True
# # #             st.success("✅ Document processed successfully!")

# # #         except Exception as e:
# # #             st.error(f"❌ Error processing document: {str(e)}")
# # #             st.stop()

# # #         # Clean up: remove the temporary file
# # #         os.remove(file_path)

# # # if st.session_state.document_processed:
# # #     # User Prompt Section
# # #     st.header("❓ Ask a Question")
# # #     prompt = st.text_input('Type your question here:', placeholder="E.g., What are the key points in this document?")

# # #     if prompt:
# # #         st.button("Generate Response")
# # #         try:
# # #             st_callback = StreamlitCallbackHandler(st.container())
# # #             with st.spinner("🤔 Generating response..."):
# # #                 @tenacity.retry(
# # #                     wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
# # #                     stop=tenacity.stop_after_attempt(5),
# # #                     reraise=True
# # #                 )
# # #                 def get_response():
# # #                     return st.session_state['agent_executor'].invoke(
# # #                         prompt,
# # #                         callbacks=[st_callback]
# # #                     )
                
# # #                 response = get_response()
            
# # #             # Check if response is a dictionary and has an 'output' key
# # #             if isinstance(response, dict) and 'output' in response:
# # #                 output = response['output']
# # #             else:
# # #                 output = str(response)
            
# # #             # Add to chat history
# # #             st.session_state.chat_history.append({"role": "user", "content": prompt})
# # #             st.session_state.chat_history.append({"role": "assistant", "content": output})
            
# # #             # Display the latest exchange
# # #             st.markdown('<div class="chat-message user"><div class="avatar">👤</div><div class="message">' + prompt + '</div></div>', unsafe_allow_html=True)
# # #             st.markdown('<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">' + output + '</div></div>', unsafe_allow_html=True)
            
# # #             # # Feedback buttons
# # #             # col1, col2, _ = st.columns([1, 1, 4])
# # #             # with col1:
# # #             #     if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.chat_history)}"):
# # #             #         st.session_state.feedback[len(st.session_state.chat_history)-1] = "positive"
# # #             #         st.success("🙏 Thank you for your feedback!")
# # #             # with col2:
# # #             #     if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.chat_history)}"):
# # #             #         st.session_state.feedback[len(st.session_state.chat_history)-1] = "negative"
# # #             #         st.error("😔 We're sorry the response wasn't helpful. We'll work on improving it!")
            
# # #         except Exception as e:
# # #             st.error(f"❌ An error occurred while processing your question: {str(e)}")

# # #     # Chat History
# # #     with st.expander("💬 View Chat History", expanded=False):
# # #         for idx, message in enumerate(st.session_state.chat_history):
# # #             if message["role"] == "user":
# # #                 st.markdown(f'<div class="chat-message user"><div class="avatar">👤</div><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
# # #             else:
# # #                 st.markdown(f'<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
# # #                 if idx in st.session_state.feedback:
# # #                     if st.session_state.feedback[idx] == "positive":
# # #                         st.markdown('👍 Helpful')
# # #                     else:
# # #                         st.markdown('👎 Not Helpful')

# # #      # Export Chat History
# # #     if st.session_state.chat_history:
# # #         st.download_button(
# # #             label="📥 Export Chat History",
# # #             data=json.dumps(st.session_state.chat_history, indent=2),
# # #             file_name="chat_history.json",
# # #             mime="application/json"
# # #         )

# # #     # Feedback Visualization
# # #     if st.session_state.feedback:
# # #         st.header("📊 Feedback Analysis")
# # #         feedback_counts = {"Helpful": 0, "Not Helpful": 0}
# # #         for feedback in st.session_state.feedback.values():
# # #             if feedback == "positive":
# # #                 feedback_counts["Helpful"] += 1
# # #             else:
# # #                 feedback_counts["Not Helpful"] += 1
        
# # #         df = pd.DataFrame(list(feedback_counts.items()), columns=['Feedback', 'Count'])
# # #         fig = px.pie(df, values='Count', names='Feedback', title='Response Feedback Distribution',
# # #                      color_discrete_map={'Helpful': '#03dac6', 'Not Helpful': '#cf6679'},
# # #                      hole=0.3)
# # #         fig.update_layout(
# # #             plot_bgcolor='rgba(0,0,0,0)',
# # #             paper_bgcolor='rgba(0,0,0,0)',
# # #             font=dict(color='#f0f0f0')
# # #         )
# # #         st.plotly_chart(fig)

# # # else:
# # #     st.info("📤 Please upload a document to begin.")

# # # # How to Use Section
# # # with st.expander("📘 How to Use OmniQuery", expanded=False):
# # #     st.markdown("""
# # #     1. **📂 Upload a Document**: Use the sidebar to upload your PDF, DOCX, TXT, or MD file.
# # #     2. **🌡️ Adjust Settings**: Set the AI temperature in the sidebar for more varied (higher) or consistent (lower) responses.
# # #     3. **❓ Ask Questions**: Type your question in the text input and press Enter.
# # #     4. **🤖 View Responses**: The AI's response will appear below your question.
# # #     5. **👍👎 Provide Feedback**: Use the thumbs up/down buttons to rate the helpfulness of responses.
# # #     6. **💬 Chat History**: Expand the 'View Chat History' section to see all interactions.
# # #     7. **📥 Export Chat**: Use the 'Export Chat History' button to download your conversation.
# # #     8. **📊 Analyze Feedback**: Check the Feedback Analysis chart to see overall response quality.
# # #     """)




# # # # Footer
# # # st.markdown("---")
# # # st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)




# # __import__('pysqlite3')
# # import sys
# # sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# # import sys
# # import tempfile
# # import os
# # import time
# # import streamlit as st
# # # from dotenv import load_dotenv
# # from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# # from langchain_community.document_loaders import (
# #     PyPDFLoader,
# #     UnstructuredFileLoader,
# #     UnstructuredWordDocumentLoader
# # )
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_chroma import Chroma
# # from langchain.agents.agent_toolkits import (
# #     create_vectorstore_agent,
# #     VectorStoreToolkit,
# #     VectorStoreInfo
# # )
# # from langchain.memory import ConversationBufferMemory
# # from langchain.callbacks.streamlit import StreamlitCallbackHandler
# # import json
# # import pandas as pd
# # import plotly.express as px
# # import tenacity

# # # Load environment variables from .env file
# # # load_dotenv()

# # # Set page configuration
# # st.set_page_config(page_title="OmniQuery", page_icon="🧠", layout="wide")

# # # Custom CSS for dark theme and improved UI
# # st.markdown("""
# # <style>
# #     # ... (CSS remains unchanged)
# # </style>
# # """, unsafe_allow_html=True)

# # # Header
# # st.markdown('<h1 style="text-align: center;">🧠 OmniQuery: Document Intelligence Platform</h1>', unsafe_allow_html=True)
# # st.markdown('<p style="text-align: center;">🚀 Upload your documents and ask any questions. OmniQuery extracts and analyzes information for you!</p>', unsafe_allow_html=True)

# # # Set up OpenAI API key from .env file
# # # openai_api_key = os.getenv("OPENAI_API_KEY")
# # openai_api_key = st.secrets['OPENAI_API_KEY']
# # if not openai_api_key:
# #     st.error("❌ OpenAI API key not found. Please check your .env file.")
# #     st.stop()

# # # Initialize OpenAI Chat model with streaming and retry logic
# # @tenacity.retry(
# #     wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
# #     stop=tenacity.stop_after_attempt(5),
# #     reraise=True
# # )
# # def get_llm(temperature):
# #     return ChatOpenAI(
# #         temperature=temperature, 
# #         model_name="gpt-3.5-turbo",
# #         streaming=True,
# #         openai_api_key=openai_api_key
# #     )

# # # Initialize embeddings with retry logic
# # @tenacity.retry(
# #     wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
# #     stop=tenacity.stop_after_attempt(5),
# #     reraise=True
# # )
# # def get_embeddings():
# #     return OpenAIEmbeddings(openai_api_key=openai_api_key)

# # # Function to load document based on file type
# # def load_document(file_path):
# #     file_extension = os.path.splitext(file_path)[1].lower()
# #     if file_extension == '.pdf':
# #         return PyPDFLoader(file_path)
# #     elif file_extension == '.docx':
# #         return UnstructuredWordDocumentLoader(file_path)
# #     elif file_extension in ['.txt', '.md']:
# #         return UnstructuredFileLoader(file_path)
# #     else:
# #         raise ValueError(f"Unsupported file type: {file_extension}")

# # # Process document function with chunking and progress bar
# # def process_document(file_path):
# #     loader = load_document(file_path)
# #     documents = loader.load()
    
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# #     texts = text_splitter.split_documents(documents)
    
# #     embeddings = get_embeddings()
    
# #     progress_bar = st.progress(0)
# #     store = None
# #     chunk_size = 100  # Process 100 texts at a time
    
# #     for i in range(0, len(texts), chunk_size):
# #         chunk = texts[i:i+chunk_size]
# #         if store is None:
# #             store = Chroma.from_documents(chunk, embeddings, collection_name='document_store')
# #         else:
# #             store.add_documents(chunk)
# #         progress = min((i + chunk_size) / len(texts), 1.0)
# #         progress_bar.progress(progress)
    
# #     progress_bar.empty()
# #     return store

# # # Initialize session state
# # if 'chat_history' not in st.session_state:
# #     st.session_state.chat_history = []
# # if 'feedback' not in st.session_state:
# #     st.session_state.feedback = {}
# # if 'document_processed' not in st.session_state:
# #     st.session_state.document_processed = False

# # # Sidebar for file upload and options
# # with st.sidebar:
# #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px">', unsafe_allow_html=True)
# #     st.header("Document Upload 📂")
# #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px">', unsafe_allow_html=True)
# #     uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt', 'md'])
    

# #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px">', unsafe_allow_html=True)
# #     st.header("Options ⚙️")
# #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px">', unsafe_allow_html=True)
# #     temperature = st.slider("🌡️ AI Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# #     # Feedback Section
# #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px;">', unsafe_allow_html=True)
# #     st.header('Feedback 📝')
# #     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px;">', unsafe_allow_html=True)
# #     st.sidebar.markdown('<p>We value your feedback! 😊</p>', unsafe_allow_html=True)

# #     feedback = st.sidebar.slider("How helpful is this tool? 😞😐😊", 1, 5, 3)
# #     feedback_text = st.sidebar.text_area("Additional feedback:")
# #     feedback_button = st.sidebar.button("Submit Feedback")
# #     if feedback_button:
# #         st.sidebar.success("Thank you for your feedback! 👍")
    
# #     if uploaded_file:
# #         st.success(f"✅ Uploaded: {uploaded_file.name}")

# # # Main content area
# # if uploaded_file is not None and not st.session_state.document_processed:
# #     # Save uploaded file temporarily
# #     with st.spinner("🔄 Processing document..."):
# #         temp_dir = tempfile.mkdtemp()
# #         if not os.path.exists(temp_dir):
# #             os.makedirs(temp_dir)
# #         file_path = os.path.join(temp_dir, uploaded_file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(uploaded_file.getbuffer())

# #         try:
# #             store = process_document(file_path)

# #             # Create vectorstore info object
# #             vectorstore_info = VectorStoreInfo(
# #                 name='document_store',
# #                 description='Processed document for Q&A',
# #                 vectorstore=store
# #             )

# #             # Add the memory to the agent
# #             memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# #             # Create the LLM
# #             llm = get_llm(temperature)

# #             # Convert the document store into a langchain toolkit
# #             toolkit = VectorStoreToolkit(
# #                 vectorstore_info=vectorstore_info,
# #                 llm=llm
# #             )

# #             # Add the toolkit to an end-to-end LC
# #             agent_executor = create_vectorstore_agent(
# #                 llm=llm,
# #                 toolkit=toolkit,
# #                 verbose=True,
# #                 agent_kwargs={"memory": memory}
# #             )

# #             st.session_state['agent_executor'] = agent_executor
# #             st.session_state['memory'] = memory
# #             st.session_state.document_processed = True
# #             st.success("✅ Document processed successfully!")

# #         except Exception as e:
# #             st.error(f"❌ Error processing document: {str(e)}")
# #             st.stop()

# #         # Clean up: remove the temporary file
# #         os.remove(file_path)

# # if st.session_state.document_processed:
# #     # User Prompt Section
# #     st.header("❓ Ask a Question")
# #     prompt = st.text_input('Type your question here:', placeholder="E.g., What are the key points in this document?")

# #     if prompt:
# #         st.button("Generate Response")
# #         try:
# #             st_callback = StreamlitCallbackHandler(st.container())
# #             with st.spinner("🤔 Generating response..."):
# #                 @tenacity.retry(
# #                     wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
# #                     stop=tenacity.stop_after_attempt(5),
# #                     reraise=True
# #                 )
# #                 def get_response():
# #                     return st.session_state['agent_executor'].invoke(
# #                         prompt,
# #                         callbacks=[st_callback]
# #                     )
                
# #                 response = get_response()
            
# #             # Check if response is a dictionary and has an 'output' key
# #             if isinstance(response, dict) and 'output' in response:
# #                 output = response['output']
# #             else:
# #                 output = str(response)
            
# #             # Add to chat history
# #             st.session_state.chat_history.append({"role": "user", "content": prompt})
# #             st.session_state.chat_history.append({"role": "assistant", "content": output})
            
# #             # Display the latest exchange
# #             st.markdown('<div class="chat-message user"><div class="avatar">👤</div><div class="message">' + prompt + '</div></div>', unsafe_allow_html=True)
# #             st.markdown('<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">' + output + '</div></div>', unsafe_allow_html=True)
            
# #             # Feedback buttons
# #             col1, col2, _ = st.columns([1, 1, 4])
# #             with col1:
# #                 if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.chat_history)}"):
# #                     st.session_state.feedback[len(st.session_state.chat_history)-1] = "positive"
# #                     st.success("🙏 Thank you for your feedback!")
# #             with col2:
# #                 if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.chat_history)}"):
# #                     st.session_state.feedback[len(st.session_state.chat_history)-1] = "negative"
# #                     st.error("😔 We're sorry the response wasn't helpful. We'll work on improving it!")
            
# #         except Exception as e:
# #             st.error(f"❌ An error occurred while processing your question: {str(e)}")

# #     # Chat History
# #     with st.expander("💬 View Chat History", expanded=False):
# #         for idx, message in enumerate(st.session_state.chat_history):
# #             if message["role"] == "user":
# #                 st.markdown(f'<div class="chat-message user"><div class="avatar">👤</div><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
# #             else:
# #                 st.markdown(f'<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
# #                 if idx in st.session_state.feedback:
# #                     if st.session_state.feedback[idx] == "positive":
# #                         st.markdown('👍 Helpful')
# #                     else:
# #                         st.markdown('👎 Not Helpful')

# #      # Export Chat History
# #     if st.session_state.chat_history:
# #         st.download_button(
# #             label="📥 Export Chat History",
# #             data=json.dumps(st.session_state.chat_history, indent=2),
# #             file_name="chat_history.json",
# #             mime="application/json"
# #         )

# #     # Feedback Visualization
# #     if st.session_state.feedback:
# #         st.header("📊 Feedback Analysis")
# #         feedback_counts = {"Helpful": 0, "Not Helpful": 0}
# #         for feedback in st.session_state.feedback.values():
# #             if feedback == "positive":
# #                 feedback_counts["Helpful"] += 1
# #             else:
# #                 feedback_counts["Not Helpful"] += 1
        
# #         df = pd.DataFrame(list(feedback_counts.items()), columns=['Feedback', 'Count'])
# #         fig = px.pie(df, values='Count', names='Feedback', title='Response Feedback Distribution',
# #                      color_discrete_map={'Helpful': '#03dac6', 'Not Helpful': '#cf6679'},
# #                      hole=0.3)
# #         fig.update_layout(
# #             plot_bgcolor='rgba(0,0,0,0)',
# #             paper_bgcolor='rgba(0,0,0,0)',
# #             font=dict(color='#f0f0f0')
# #         )
# #         st.plotly_chart(fig)

# # else:
# #     st.info("📤 Please upload a document to begin.")

# # # How to Use Section
# # with st.expander("📘 How to Use OmniQuery", expanded=False):
# #     st.markdown("""
# #     1. **📂 Upload a Document**: Use the sidebar to upload your PDF, DOCX, TXT, or MD file.
# #     2. **🌡️ Adjust Settings**: Set the AI temperature in the sidebar for more varied (higher) or consistent (lower) responses.
# #     3. **❓ Ask Questions**: Type your question in the text input and press Enter.
# #     4. **🤖 View Responses**: The AI's response will appear below your question.
# #     5. **👍👎 Provide Feedback**: Use the thumbs up/down buttons to rate the helpfulness of responses.
# #     6. **💬 Chat History**: Expand the 'View Chat History' section to see all interactions.
# #     7. **📥 Export Chat**: Use the 'Export Chat History' button to download your conversation.
# #     8. **📊 Analyze Feedback**: Check the Feedback Analysis chart to see overall response quality.
# #     """)

# # # Footer
# # st.markdown("---")
# # st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)























































# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# import sys
# import tempfile
# import os
# import time
# import streamlit as st
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.document_loaders import (
#     PyPDFLoader,
#     UnstructuredFileLoader,
#     UnstructuredWordDocumentLoader
# )
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain.agents.agent_toolkits import (
#     create_vectorstore_agent,
#     VectorStoreToolkit,
#     VectorStoreInfo
# )
# from langchain.memory import ConversationBufferMemory
# from langchain.callbacks.streamlit import StreamlitCallbackHandler
# import json
# import pandas as pd
# import plotly.express as px
# import tenacity

# # Set page configuration
# st.set_page_config(page_title="OmniQuery", page_icon="🧠", layout="wide")

# # Custom CSS for dark theme and improved UI
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;700&family=Inter:wght@300;400;600&display=swap');
    
#     :root {
#         --primary-color: #00F5FF;
#         --secondary-color: #FF00E4;
#         --bg-color: #0A0E17;
#         --text-color: #E0E0E0;
#         --card-bg: #141C2F;
#     }
    
#     body {
#         color: var(--text-color);
#         background-color: var(--bg-color);
#         font-family: 'Inter', sans-serif;
#         background-image: 
#             radial-gradient(circle at 10% 20%, rgba(0, 245, 255, 0.1) 0%, transparent 20%),
#             radial-gradient(circle at 90% 80%, rgba(255, 0, 228, 0.1) 0%, transparent 20%);
#         background-attachment: fixed;
#     }
    
#     .stApp {
#         background: transparent;
#     }
    
#     h1, h2, h3 {
#         font-family: 'Exo 2', sans-serif;
#         color: var(--primary-color);
#         text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
#         letter-spacing: 1px;
#     }
    
#     .stButton > button {
#         font-family: 'Exo 2', sans-serif;
#         background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
#         color: var(--bg-color);
#         font-weight: 700;
#         border-radius: 30px;
#         border: none;
#         padding: 15px 30px;
#         font-size: 16px;
#         transition: all 0.3s ease;
#         box-shadow: 0 5px 15px rgba(0, 245, 255, 0.4);
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-3px) scale(1.05);
#         box-shadow: 0 8px 20px rgba(255, 0, 228, 0.6);
#     }
    
#     .stTextInput > div > div > input, 
#     .stSelectbox > div > div > select, 
#     .stTextArea > div > div > textarea {
#         font-family: 'Inter', sans-serif;
#         background-color: var(--card-bg);
#         color: var(--text-color);
#         border-radius: 15px;
#         border: 2px solid var(--primary-color);
#         padding: 12px;
#         transition: all 0.3s ease;
#     }
    
#     .stTextInput > div > div > input:focus, 
#     .stSelectbox > div > div > select:focus, 
#     .stTextArea > div > div > textarea:focus {
#         border-color: var(--secondary-color);
#         box-shadow: 0 0 15px rgba(255, 0, 228, 0.5);
#     }
    
#     .stTab {
#         font-family: 'Exo 2', sans-serif;
#         background-color: var(--card-bg);
#         color: var(--text-color);
#         font-weight: 600;
#         border-radius: 10px 10px 0 0;
#         border: 2px solid var(--primary-color);
#         border-bottom: none;
#         transition: all 0.3s ease;
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
    
#     .stTab[aria-selected="true"] {
#         background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
#         color: var(--bg-color);
#     }
    
#     .stDataFrame {
#         font-family: 'Inter', sans-serif;
#         border: 2px solid var(--primary-color);
#         border-radius: 15px;
#         overflow: hidden;
#     }
    
#     .stDataFrame thead {
#         background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
#         color: var(--bg-color);
#         font-family: 'Exo 2', sans-serif;
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
    
#     .stDataFrame tbody tr:nth-of-type(even) {
#         background-color: rgba(20, 28, 47, 0.7);
#     }
    
#     .stAlert {
#         font-family: 'Inter', sans-serif;
#         background-color: var(--card-bg);
#         color: var(--text-color);
#         border-radius: 15px;
#         border: 2px solid var(--primary-color);
#     }
    
#     .stProgress > div > div > div > div {
#         background-color: var(--primary-color);
#     }
    
#     .stSlider > div > div > div > div {
#         color: var(--primary-color);
#         font-family: 'Exo 2', sans-serif;
#     }
    
#     .css-1cpxqw2 {
#         background-color: var(--card-bg);
#         border-radius: 20px;
#         padding: 25px;
#         box-shadow: 0 10px 30px rgba(0, 245, 255, 0.2);
#         transition: all 0.3s ease;
#         border: 2px solid transparent;
#         background-clip: padding-box;
#     }
    
#     .css-1cpxqw2:hover {
#         transform: translateY(-5px);
#         box-shadow: 0 15px 35px rgba(255, 0, 228, 0.3);
#         border-color: var(--secondary-color);
#     }
    
#     @keyframes glow {
#         0% { box-shadow: 0 0 5px var(--primary-color); }
#         50% { box-shadow: 0 0 20px var(--primary-color), 0 0 30px var(--secondary-color); }
#         100% { box-shadow: 0 0 5px var(--primary-color); }
#     }
    
#     .glow-effect {
#         animation: glow 2s infinite;
#     }

#     a {
#         color: var(--primary-color);
#         text-decoration: none;
#         transition: all 0.3s ease;
#         position: relative;
#     }

#     a:hover {
#         color: var(--secondary-color);
#     }

#     a::after {
#         content: '';
#         position: absolute;
#         width: 100%;
#         height: 2px;
#         bottom: -2px;
#         left: 0;
#         background-color: var(--secondary-color);
#         visibility: hidden;
#         transform: scaleX(0);
#         transition: all 0.3s ease-in-out;
#     }

#     a:hover::after {
#         visibility: visible;
#         transform: scaleX(1);
#     }
# </style>
# """, unsafe_allow_html=True)

# # Header
# st.markdown('<h1 style="text-align: center;">🧠 OmniQuery: Document Intelligence Platform</h1>', unsafe_allow_html=True)
# st.markdown('<p style="text-align: center;">🚀 Upload your documents and ask any questions. OmniQuery extracts and analyzes information for you!</p>', unsafe_allow_html=True)

# # Set up OpenAI API key from .env file
# openai_api_key = st.secrets['OPENAI_API_KEY']
# if not openai_api_key:
#     st.error("❌ OpenAI API key not found. Please check your .env file.")
#     st.stop()

# # Initialize OpenAI Chat model with streaming and retry logic
# @tenacity.retry(
#     wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
#     stop=tenacity.stop_after_attempt(5),
#     reraise=True
# )
# def get_llm(temperature):
#     return ChatOpenAI(
#         temperature=temperature, 
#         model_name="gpt-3.5-turbo",
#         streaming=True,
#         openai_api_key=openai_api_key
#     )

# # Initialize embeddings with retry logic
# @tenacity.retry(
#     wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
#     stop=tenacity.stop_after_attempt(5),
#     reraise=True
# )
# def get_embeddings():
#     return OpenAIEmbeddings(openai_api_key=openai_api_key)

# # Function to load document based on file type
# def load_document(file_path):
#     file_extension = os.path.splitext(file_path)[1].lower()
#     if file_extension == '.pdf':
#         return PyPDFLoader(file_path)
#     elif file_extension == '.docx':
#         return UnstructuredWordDocumentLoader(file_path)
#     elif file_extension in ['.txt', '.md']:
#         return UnstructuredFileLoader(file_path)
#     else:
#         raise ValueError(f"Unsupported file type: {file_extension}")

# # Process document function with chunking and progress bar
# def process_document(file_path):
#     loader = load_document(file_path)
#     documents = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_documents(documents)
    
#     embeddings = get_embeddings()
    
#     progress_bar = st.progress(0)
#     store = None
#     chunk_size = 100  # Process 100 texts at a time
    
#     for i in range(0, len(texts), chunk_size):
#         chunk = texts[i:i+chunk_size]
#         if store is None:
#             store = Chroma.from_documents(chunk, embeddings, collection_name='document_store')
#         else:
#             store.add_documents(chunk)
#         progress = min((i + chunk_size) / len(texts), 1.0)
#         progress_bar.progress(progress)
    
#     progress_bar.empty()
#     return store

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'feedback' not in st.session_state:
#     st.session_state.feedback = {}
# if 'document_processed' not in st.session_state:
#     st.session_state.document_processed = False

# # Sidebar for file upload and options
# with st.sidebar:
#     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px">', unsafe_allow_html=True)
#     st.header("Document Upload 📂")
#     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px">', unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt', 'md'])
    

#     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px">', unsafe_allow_html=True)
#     st.header("Options ⚙️")
#     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px">', unsafe_allow_html=True)
#     temperature = st.slider("🌡️ AI Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

#     # Feedback Section
#     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px;">', unsafe_allow_html=True)
#     st.header('Feedback 📝')
#     st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px;">', unsafe_allow_html=True)
#     st.sidebar.markdown('<p>We value your feedback! 😊</p>', unsafe_allow_html=True)

#     feedback = st.sidebar.slider("How helpful is this tool? 😞😐😊", 1, 5, 3)
#     feedback_text = st.sidebar.text_area("Additional feedback:")
#     feedback_button = st.sidebar.button("Submit Feedback")
#     if feedback_button:
#         st.sidebar.success("Thank you for your feedback! 👍")
    
#     if uploaded_file:
#         st.success(f"✅ Uploaded: {uploaded_file.name}")

# # Main content area
# if uploaded_file is not None and not st.session_state.document_processed:
#     # Save uploaded file temporarily
#     with st.spinner("🔄 Processing document..."):
#         temp_dir = tempfile.mkdtemp()
#         if not os.path.exists(temp_dir):
#             os.makedirs(temp_dir)
#         file_path = os.path.join(temp_dir, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         try:
#             store = process_document(file_path)

#             # Create vectorstore info object
#             vectorstore_info = VectorStoreInfo(
#                 name='document_store',
#                 description='Processed document for Q&A',
#                 vectorstore=store
#             )

#             # Add the memory to the agent
#             memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#             # Create the LLM
#             llm = get_llm(temperature)

#             # Convert the document store into a langchain toolkit
#             toolkit = VectorStoreToolkit(
#                 vectorstore_info=vectorstore_info,
#                 llm=llm
#             )

#             # Add the toolkit to an end-to-end LC
#             agent_executor = create_vectorstore_agent(
#                 llm=llm,
#                 toolkit=toolkit,
#                 verbose=True,
#                 agent_kwargs={"memory": memory}
#             )

#             st.session_state['agent_executor'] = agent_executor
#             st.session_state['memory'] = memory
#             st.session_state.document_processed = True
#             st.success("✅ Document processed successfully!")

#         except Exception as e:
#             st.error(f"❌ Error processing document: {str(e)}")
#             st.stop()

#         # Clean up: remove the temporary file
#         os.remove(file_path)

# if st.session_state.document_processed:
#     # User Prompt Section
#     st.header("❓ Ask a Question")
# prompt = st.text_input('Type your question here:', placeholder="E.g., What are the key points in this document?")

# if prompt:
#     st.button("Generate Response")
#     try:
#         st_callback = StreamlitCallbackHandler(st.container())
#         with st.spinner("🤔 Generating response..."):
#             @tenacity.retry(
#                 wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
#                 stop=tenacity.stop_after_attempt(5),
#                 reraise=True
#             )
#             def get_response():
#                 return st.session_state['agent_executor'].invoke(
#                     prompt,
#                     callbacks=[st_callback]
#                 )
            
#             response = get_response()
        
#         # Check if response is a dictionary and has an 'output' key
#         if isinstance(response, dict) and 'output' in response:
#             output = response['output']
#         else:
#             output = str(response)
        
#         # Add to chat history
#         st.session_state.chat_history.append({"role": "user", "content": prompt})
#         st.session_state.chat_history.append({"role": "assistant", "content": output})
        
#         # Display the latest exchange
#         st.markdown('<div class="chat-message user"><div class="avatar">👤</div><div class="message">' + prompt + '</div></div>', unsafe_allow_html=True)
#         st.markdown('<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">' + output + '</div></div>', unsafe_allow_html=True)
        
#         # Feedback buttons
#         col1, col2, _ = st.columns([1, 1, 4])
#         with col1:
#             if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.chat_history)}"):
#                 st.session_state.feedback[len(st.session_state.chat_history)-1] = "positive"
#                 st.success("🙏 Thank you for your feedback!")
#         with col2:
#             if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.chat_history)}"):
#                 st.session_state.feedback[len(st.session_state.chat_history)-1] = "negative"
#                 st.error("😔 We're sorry the response wasn't helpful. We'll work on improving it!")
        
#     except Exception as e:
#         st.error(f"❌ An error occurred while processing your question: {str(e)}")

# # Chat History
# with st.expander("💬 View Chat History", expanded=False):
#     for idx, message in enumerate(st.session_state.chat_history):
#         if message["role"] == "user":
#             st.markdown(f'<div class="chat-message user"><div class="avatar">👤</div><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
#             if idx in st.session_state.feedback:
#                 if st.session_state.feedback[idx] == "positive":
#                     st.markdown('👍 Helpful')
#                 else:
#                     st.markdown('👎 Not Helpful')

# # Export Chat History
# if st.session_state.chat_history:
#     st.download_button(
#         label="📥 Export Chat History",
#         data=json.dumps(st.session_state.chat_history, indent=2),
#         file_name="chat_history.json",
#         mime="application/json"
#     )

# # Feedback Visualization
# if st.session_state.feedback:
#     st.header("📊 Feedback Analysis")
#     feedback_counts = {"Helpful": 0, "Not Helpful": 0}
#     for feedback in st.session_state.feedback.values():
#         if feedback == "positive":
#             feedback_counts["Helpful"] += 1
#         else:
#             feedback_counts["Not Helpful"] += 1
    
#     df = pd.DataFrame(list(feedback_counts.items()), columns=['Feedback', 'Count'])
#     fig = px.pie(df, values='Count', names='Feedback', title='Response Feedback Distribution',
#                     color_discrete_map={'Helpful': '#03dac6', 'Not Helpful': '#cf6679'},
#                     hole=0.3)
#     fig.update_layout(
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         font=dict(color='#f0f0f0')
#     )
#     st.plotly_chart(fig)

# else:
#     st.info("📤 Please upload a document to begin.")

# # How to Use Section
# with st.expander("📘 How to Use OmniQuery", expanded=False):
#     st.markdown("""
#     1. **📂 Upload a Document**: Use the sidebar to upload your PDF, DOCX, TXT, or MD file.
#     2. **🌡️ Adjust Settings**: Set the AI temperature in the sidebar for more varied (higher) or consistent (lower) responses.
#     3. **❓ Ask Questions**: Type your question in the text input and press Enter.
#     4. **🤖 View Responses**: The AI's response will appear below your question.
#     5. **👍👎 Provide Feedback**: Use the thumbs up/down buttons to rate the helpfulness of responses.
#     6. **💬 Chat History**: Expand the 'View Chat History' section to see all interactions.
#     7. **📥 Export Chat**: Use the 'Export Chat History' button to download your conversation.
#     8. **📊 Analyze Feedback**: Check the Feedback Analysis chart to see overall response quality.
#     """)

# # Footer
# st.markdown("---")
# st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)





# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sys
import tempfile
import os
import time
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streamlit import StreamlitCallbackHandler
import json
import pandas as pd
import plotly.express as px
import tenacity
import base64


# Set page configuration
st.set_page_config(page_title="OmniQuery", page_icon="🧠", layout="wide")

# Custom CSS for dark theme and improved UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;700&family=Inter:wght@300;400;600&display=swap');
    
    :root {
        --primary-color: #00F5FF;
        --secondary-color: #FF00E4;
        --bg-color: #0A0E17;
        --text-color: #E0E0E0;
        --card-bg: #141C2F;
    }
    
    body {
        color: var(--text-color);
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 245, 255, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(255, 0, 228, 0.1) 0%, transparent 20%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: transparent;
    }
    
    h1, h2, h3 {
        font-family: 'Exo 2', sans-serif;
        color: var(--primary-color);
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
        letter-spacing: 1px;
    }
    
    .stButton > button {
        font-family: 'Exo 2', sans-serif;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: var(--bg-color);
        font-weight: 700;
        border-radius: 30px;
        border: none;
        padding: 15px 30px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 245, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(255, 0, 228, 0.6);
    }
    
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > select, 
    .stTextArea > div > div > textarea {
        font-family: 'Inter', sans-serif;
        background-color: var(--card-bg);
        color: var(--text-color);
        border-radius: 15px;
        border: 2px solid var(--primary-color);
        padding: 12px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus, 
    .stSelectbox > div > div > select:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: var(--secondary-color);
        box-shadow: 0 0 15px rgba(255, 0, 228, 0.5);
    }
    
    .stTab {
        font-family: 'Exo 2', sans-serif;
        background-color: var(--card-bg);
        color: var(--text-color);
        font-weight: 600;
        border-radius: 10px 10px 0 0;
        border: 2px solid var(--primary-color);
        border-bottom: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stTab[aria-selected="true"] {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: var(--bg-color);
    }
    
    .stDataFrame {
        font-family: 'Inter', sans-serif;
        border: 2px solid var(--primary-color);
        border-radius: 15px;
        overflow: hidden;
    }
    
    .stDataFrame thead {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: var(--bg-color);
        font-family: 'Exo 2', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stDataFrame tbody tr:nth-of-type(even) {
        background-color: rgba(20, 28, 47, 0.7);
    }
    
    .stAlert {
        font-family: 'Inter', sans-serif;
        background-color: var(--card-bg);
        color: var(--text-color);
        border-radius: 15px;
        border: 2px solid var(--primary-color);
    }
    
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }
    
    .stSlider > div > div > div > div {
        color: var(--primary-color);
        font-family: 'Exo 2', sans-serif;
    }
    
    .css-1cpxqw2 {
        background-color: var(--card-bg);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0, 245, 255, 0.2);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    
    .css-1cpxqw2:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(255, 0, 228, 0.3);
        border-color: var(--secondary-color);
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px var(--primary-color); }
        50% { box-shadow: 0 0 20px var(--primary-color), 0 0 30px var(--secondary-color); }
        100% { box-shadow: 0 0 5px var(--primary-color); }
    }
    
    .glow-effect {
        animation: glow 2s infinite;
    }

    a {
        color: var(--primary-color);
        text-decoration: none;
        transition: all 0.3s ease;
        position: relative;
    }

    a:hover {
        color: var(--secondary-color);
    }

    a::after {
        content: '';
        position: absolute;
        width: 100%;
        height: 2px;
        bottom: -2px;
        left: 0;
        background-color: var(--secondary-color);
        visibility: hidden;
        transform: scaleX(0);
        transition: all 0.3s ease-in-out;
    }

    a:hover::after {
        visibility: visible;
        transform: scaleX(1);
    }
            



</style>
""", unsafe_allow_html=True)

## Header with logo
st.markdown(
    '''
    <h1 style="text-align: center;">
        <img src="data:image/png;base64,{}" width="150" />
        OmniQuery: Document Intelligence Platform
    </h1>
    '''.format(base64.b64encode(open("New_logo.png", "rb").read()).decode('utf-8')),
    unsafe_allow_html=True
)

st.markdown('<p style="text-align: center;">🚀 Upload your documents and ask any questions. OmniQuery extracts and analyzes information for you!</p>', unsafe_allow_html=True)

# Set up OpenAI API key from .env file
openai_api_key = st.secrets['OPENAI_API_KEY']
if not openai_api_key:
    st.error("❌ OpenAI API key not found. Please check your .env file.")
    st.stop()

# Initialize OpenAI Chat model with streaming and retry logic
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(5),
    reraise=True
)
def get_llm(temperature):
    return ChatOpenAI(
        temperature=temperature, 
        model_name="gpt-3.5-turbo",
        streaming=True,
        openai_api_key=openai_api_key
    )

# Initialize embeddings with retry logic
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(5),
    reraise=True
)
def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=openai_api_key)

# Function to load document based on file type
def load_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension == '.docx':
        return UnstructuredWordDocumentLoader(file_path)
    elif file_extension in ['.txt', '.md']:
        return UnstructuredFileLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

# Process document function with chunking and progress bar
def process_document(file_path):
    loader = load_document(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings()
    
    progress_bar = st.progress(0)
    store = None
    chunk_size = 100  # Process 100 texts at a time
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        if store is None:
            store = Chroma.from_documents(chunk, embeddings, collection_name='document_store')
        else:
            store.add_documents(chunk)
        progress = min((i + chunk_size) / len(texts), 1.0)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    return store

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

# Sidebar for file upload and options
with st.sidebar:

    # st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px">', unsafe_allow_html=True)
    st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px">', unsafe_allow_html=True)
    st.header("Document Upload 📂")
    st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt', 'md'])
    

    st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px">', unsafe_allow_html=True)
    st.header("Options ⚙️")
    st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px">', unsafe_allow_html=True)
    temperature = st.slider("🌡️ AI Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    # Feedback Section
    st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-bottom:2px;">', unsafe_allow_html=True)
    st.header('Feedback 📝')
    st.sidebar.markdown('<hr style="border:1px solid #FFD700; margin-top:2px;">', unsafe_allow_html=True)
    st.sidebar.markdown('<p>We value your feedback! 😊</p>', unsafe_allow_html=True)

    feedback = st.sidebar.slider("How helpful is this tool? 😞😐😊", 1, 5, 3)
    feedback_text = st.sidebar.text_area("Additional feedback:")
    feedback_button = st.sidebar.button("Submit Feedback")
    if feedback_button:
        st.sidebar.success("Thank you for your feedback! 👍")
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")

# Main content area
if uploaded_file is not None and not st.session_state.document_processed:
    # Save uploaded file temporarily
    with st.spinner("🔄 Processing document..."):
        temp_dir = tempfile.mkdtemp()
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            store = process_document(file_path)

            # Create vectorstore info object
            vectorstore_info = VectorStoreInfo(
                name='document_store',
                description='Processed document for Q&A',
                vectorstore=store
            )

            # Add the memory to the agent
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # Create the LLM
            llm = get_llm(temperature)

            # Convert the document store into a langchain toolkit
            toolkit = VectorStoreToolkit(
                vectorstore_info=vectorstore_info,
                llm=llm
            )

            # Add the toolkit to an end-to-end LC
            agent_executor = create_vectorstore_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_kwargs={"memory": memory}
            )

            st.session_state['agent_executor'] = agent_executor
            st.session_state['memory'] = memory
            st.session_state.document_processed = True
            st.success("✅ Document processed successfully!")

        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            st.stop()

        # Clean up: remove the temporary file
        os.remove(file_path)

if st.session_state.document_processed:
    # User Prompt Section
    st.header("❓ Ask a Question")
prompt = st.text_input('Type your question here:', placeholder="E.g., What are the key points in this document?")

if prompt:
    st.button("Generate Response")
    try:
        st_callback = StreamlitCallbackHandler(st.container())
        with st.spinner("🤔 Generating response..."):
            @tenacity.retry(
                wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                stop=tenacity.stop_after_attempt(5),
                reraise=True
            )
            def get_response():
                return st.session_state['agent_executor'].invoke(
                    prompt,
                    callbacks=[st_callback]
                )
            
            response = get_response()
        
        # Check if response is a dictionary and has an 'output' key
        if isinstance(response, dict) and 'output' in response:
            output = response['output']
        else:
            output = str(response)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": output})
        
        # Display the latest exchange
        st.markdown('<div class="chat-message user"><div class="avatar">👤</div><div class="message">' + prompt + '</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">' + output + '</div></div>', unsafe_allow_html=True)
        
        # Feedback buttons
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.chat_history)}"):
                st.session_state.feedback[len(st.session_state.chat_history)-1] = "positive"
                st.success("🙏 Thank you for your feedback!")
        with col2:
            if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.chat_history)}"):
                st.session_state.feedback[len(st.session_state.chat_history)-1] = "negative"
                st.error("😔 We're sorry the response wasn't helpful. We'll work on improving it!")
        
    except Exception as e:
        st.error(f"❌ An error occurred while processing your question: {str(e)}")

# Chat History
with st.expander("💬 View Chat History", expanded=False):
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user"><div class="avatar">👤</div><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
            if idx in st.session_state.feedback:
                if st.session_state.feedback[idx] == "positive":
                    st.markdown('👍 Helpful')
                else:
                    st.markdown('👎 Not Helpful')

# Export Chat History
if st.session_state.chat_history:
    st.download_button(
        label="📥 Export Chat History",
        data=json.dumps(st.session_state.chat_history, indent=2),
        file_name="chat_history.json",
        mime="application/json"
    )

# Feedback Visualization
if st.session_state.feedback:
    st.header("📊 Feedback Analysis")
    feedback_counts = {"Helpful": 0, "Not Helpful": 0}
    for feedback in st.session_state.feedback.values():
        if feedback == "positive":
            feedback_counts["Helpful"] += 1
        else:
            feedback_counts["Not Helpful"] += 1
    
    df = pd.DataFrame(list(feedback_counts.items()), columns=['Feedback', 'Count'])
    fig = px.pie(df, values='Count', names='Feedback', title='Response Feedback Distribution',
                    color_discrete_map={'Helpful': '#03dac6', 'Not Helpful': '#cf6679'},
                    hole=0.3)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f0f0f0')
    )
    st.plotly_chart(fig)

else:
    st.info("📤 Please upload a document to begin.")

# How to Use Section
with st.expander("📘 How to Use OmniQuery", expanded=False):
    st.markdown("""
    1. **📂 Upload a Document**: Use the sidebar to upload your PDF, DOCX, TXT, or MD file.
    2. **🌡️ Adjust Settings**: Set the AI temperature in the sidebar for more varied (higher) or consistent (lower) responses.
    3. **❓ Ask Questions**: Type your question in the text input and press Enter.
    4. **🤖 View Responses**: The AI's response will appear below your question.
    5. **👍👎 Provide Feedback**: Use the thumbs up/down buttons to rate the helpfulness of responses.
    6. **💬 Chat History**: Expand the 'View Chat History' section to see all interactions.
    7. **📥 Export Chat**: Use the 'Export Chat History' button to download your conversation.
    8. **📊 Analyze Feedback**: Check the Feedback Analysis chart to see overall response quality.
    """)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)
