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
# Title of app
st.title("Chat with Your Italian Buddy!")

# Initialize OpenAI client with API key from Streamlit secrets
client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini")


file_path = "italy_book.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)
faiss_index = FAISS.from_documents(splits, OpenAIEmbeddings())



# Initialize or retrieve the conversation history from session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Handle new user input
user_input = st.chat_input("Type your message...")
if user_input:
    # Append user message to the conversation history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    embedding_vector = OpenAIEmbeddings().embed_query(user_input)
    sources = faiss_index.similarity_search_by_vector(embedding_vector, k=2)

    template = f"\n Only using the following documents and cite the resource at the end: "
    for source in sources:
        template += "\n"+source.page_content.strip()
    template += """Question: {Question}
    Answer: """
    combined_prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(llm=llm, prompt=combined_prompt)

    # Get response from GPT-4
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]],
    # )
    
    # # Extract the text from the response
    # assistant_response = response.choices[0].message.content
    response = llm_chain.invoke(user_input)

    # Append assistant's response to the conversation history
    st.session_state["messages"].append({"role": "assistant", "content": response["text"]})

# Display previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
