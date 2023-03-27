import pymongo
import pinecone
import streamlit as st
import os

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["internal-docs"]
mycol = mydb["documents"]


st.markdown("### :blue[Upload Data]")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Upload From Local PDF File", "Upload From Online PDF File", "Upload From Notion", "Upload From Website"])

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

with tab1:
    uploaded_file = st.file_uploader("Choose a file", type="pdf")
    if st.button('Upload Document'):
        document_progress = st.empty()
        if uploaded_file is not None:
            with st.spinner('Uploading document...'):
                loader = UnstructuredPDFLoader(uploaded_file)
                data = loader.load()
                document = {
                    "name": uploaded_file.name,
                    "content": data[0].page_content
                }
                mycol.insert_one(document)
        else:
            st.error("No file uploaded")

with tab2:
    link_to_pdf = st.text_input("Enter link to PDF")
    if st.button('Upload PDF Link'):
        if link_to_pdf is not None:
            with st.spinner('Uploading document...'):
                loader = UnstructuredPDFLoader(link_to_pdf)
                data = loader.load()
                document = {
                    "name": link_to_pdf,
                    "content": data[0].page_content
                }
                mycol.insert_one(document)
        else:
            st.error("No link entered")

with tab3:
    st.write("Notion")

with tab4:
    link_to_website = st.text_input("Enter link to website")
    if st.button('Upload Website Link'):
        if link_to_website is not None:
            st.write(link_to_website)
        else:
            st.error("No link entered")


st.markdown("### :blue[Question]")
question = st.text_input("Ask a question about the document")

if st.button('Ask Question'):
    with st.spinner('Searching for answer...'):
        while True:
            try:
                # getting the documents and splitting them into chunks
                x = mycol.find({})
                content = ''
                for i in x:
                    content += i['content']
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=0)
                formated_document = text_splitter.create_documents(
                    [x['content']])
                texts = text_splitter.split_documents(formated_document)
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                # search for most relevant docs
                pinecone.init(
                    api_key=PINECONE_API_KEY,
                    environment=PINECONE_API_ENV
                )
                index_name = "internaldoc"
                docsearch = Pinecone.from_texts(
                    [t.page_content for t in texts], embeddings, index_name=index_name)
                query = question
                docs = docsearch.similarity_search(
                    query, include_metadata=True)
                # use the most relevant doc to answer the question
                llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
                chain = load_qa_chain(llm, chain_type="stuff")
                result = chain.run(input_documents=docs, question=query)
                st.write(result)
                break
            except ValueError:
                st.write(ValueError)
