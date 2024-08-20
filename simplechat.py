import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Title of the app
st.title("Chat with Your Italian Buddy!")

# Initialize OpenAI client with API key from Streamlit secrets
client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini")

file_path = "italy_book.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)

# Initialize or retrieve embeddings and FAISS index
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = OpenAIEmbeddings()
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = FAISS.from_documents(splits, st.session_state["embeddings"])

# Initialize or retrieve the conversation history from session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Handle new user input
user_input = st.chat_input("Type your message...")
if user_input:
    # Append user message to the conversation history
    st.session_state["messages"].append({"role": "user", "content": user_input})


#faiss_index is the source(book) || embedding vector is the user input
    embedding_vector = st.session_state["embeddings"].embed_query(user_input)
    sources = st.session_state["faiss_index"].similarity_search_by_vector(embedding_vector, k=2)

    template = "\nOnly use the following documents, do not answer any questions not italy related, allow talking in multiple languages if the user siwtched language but keep the defaut english\n"
    for source in sources:
        template += "\n" + source.page_content.strip()
    template += "\nQuestion: {Question}\nAnswer: "

    combined_prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(llm=llm, prompt=combined_prompt)

    # Use conversation history in the prompt
    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["messages"]])
    full_prompt = f"{history}\n\n{user_input}"

    # Get response from the LLMChain
    response = llm_chain.invoke({"Question": full_prompt})

    # Append assistant's response to the conversation history
    st.session_state["messages"].append({"role": "assistant", "content": response["text"]})

# Display previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])