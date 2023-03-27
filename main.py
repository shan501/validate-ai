import pymongo
import pinecone
import streamlit as st
import os

from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredURLLoader, NotionDirectoryLoader
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

st.markdown("### :blue[Documents Uploaded So Far]")
uploaded_documents = ''
x = mycol.find({})

for i in x:
    uploaded_documents += "[   " + i['name'] + "   ]" + " "

st.write(uploaded_documents)

st.markdown("### :blue[Upload Data]")
tab1, tab2, tab3 = st.tabs(
    ["Upload From Locally", "Upload From Website", "Upload From Notion"])

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

with tab1:
    uploaded_file = st.file_uploader("Choose a file")
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
                document_progress.write("Document Uploaded")
        else:
            st.error("No file uploaded")

with tab2:
    link_to_pdf = st.text_input("Enter link To Website")
    if st.button('Upload Website Link'):
        document_progress = st.empty()
        if link_to_pdf is not None:
            with st.spinner('Uploading document...'):
                loader = UnstructuredURLLoader([link_to_pdf])
                data = loader.load()
                document = {
                    "name": link_to_pdf,
                    "content": data[0].page_content
                }
                mycol.insert_one(document)
                document_progress.write("Document Uploaded")
        else:
            st.error("No link entered")

with tab3:
    notion_file = st.file_uploader(
        "Please export your database from notion. You can do this by clicking on the three dots in the upper right hand corner and then clicking Export. Please select the format PDF export from the drop down.", type="pdf")
    if st.button('Upload Notion File'):
        document_progress = st.empty()
        if notion_file is not None:
            with st.spinner('Uploading document...'):
                loader = UnstructuredURLLoader(notion_file)
                data = loader.load()
                st.write(data)
                document = {
                    "name": notion_file,
                    "content": data[0].page_content
                }
                mycol.insert_one(document)
                document_progress.write("Document Uploaded")
        else:
            st.error("No link entered")


st.markdown("### :blue[Question]")
question = st.text_input("Ask a question about the document")

if st.button('Ask Question'):
    with st.spinner('Searching for answer...'):
        # getting the documents and splitting them into chunks
        x = mycol.find({})
        content = ''
        for i in x:
            content += i['content']
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0)
        formated_document = text_splitter.create_documents(
            [content])
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
        # # use the most relevant doc to answer the question
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        result = chain.run(input_documents=docs, question=query)
        st.write(result)
