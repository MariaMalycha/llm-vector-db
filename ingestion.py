from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_pinecone import PineconeVectorStore  



if __name__=="__main__":
    print('advanced rag')
    loader = TextLoader("/Users/mariamalycha/Documents/llm-programs/llm-vector-db/mediumblog.txt")
    document = loader.load()
    print("splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(document)
    print(f"created {len(text)} chunks")
    embedings = OpenAIEmbeddings(openai_api_type=os.environ.get('OPENAI_API_KEY'))
    print('injesting')
    PineconeVectorStore.from_documents(text, embedings, index_name=os.environ['PINECONE_INDEX_NAME'])
    print('finish')
