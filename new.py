import streamlit as st
import os
import torch
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


def load_document(file):
    # Determine file type and load accordingly
    name, extension = os.path.splitext(file)
    if extension == '.pdf':
        print(f'Loading {file}')
        loader = UnstructuredPDFLoader(file_path=file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f'Loading {file}')
        loader = TextLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    else:
        print('Document format is not supported')
        return None
    data = loader.load()
    return data

def chunk_data(data, chunk_size=7500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, q, llm_model="mistral"):
    # Using MultiQueryRetriever instead of similarity search
    llm = ChatOllama(model=llm_model,device=torch.device('cuda'))
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 
        different versions of the given user question to retrieve relevant documents 
        from a vector database. By generating multiple perspectives on the user question, 
        your goal is to help the user overcome some of the limitations of the distance-based 
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(),
        llm=llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke({"question": q})
    return answer


if __name__ == "__main__":
    # st.image('img.png')
    st.subheader("LLM Question Answering Application 🤖")
    answer = None
    # Sidebar for user inputs
    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk Size:', min_value=500, max_value=10000, value=7500)
        add_data = st.button('Add Data')

        if uploaded_file and add_data:
            with st.spinner('Reading, Chunking, and Embedding the data...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                # Load, chunk, and embed the data
                data = load_document(file_name)
                if data:
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    vector_store = create_embeddings(chunks)
                    st.session_state.vs = vector_store
                    st.success('Data loaded successfully')

    q = st.text_input('Ask a question about the content of your file:')
    
    # If the user has uploaded a file and asks a question
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write('Searching for an answer...')
            answer = ask_and_get_answer(vector_store, q)
            st.text_area('Answer:', value=answer, height=300)

    st.divider()
    if 'history' not in st.session_state:
        st.session_state.history = ''
    if q and answer:
        value = f'Q:{q} \n A:{answer} \n'
        st.session_state.history += f'{value} \n {"-"*100} \n {st.session_state.history}'
    h = st.session_state.history
    st.text_area(label ='History:', value=h, key='history',height=400)