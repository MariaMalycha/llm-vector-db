from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_pinecone import PineconeVectorStore  
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.vectorstores import FAISS

if __name__=="__main__":
    print("retriving")
    embedings = OpenAIEmbeddings()
    llm = ChatOpenAI()
    query = "what is Pinecone in machine learing"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)
    
    vectrostore = PineconeVectorStore(index_name=os.environ['PINECONE_INDEX_NAME'],
                                      embedding=embedings

    )

    retrival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrival_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectrostore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = retrival_chain.invoke(input={'input': query})
    print(result)



