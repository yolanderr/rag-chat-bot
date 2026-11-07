import langchain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from schema import StartupReview

load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")


#Langchain Expression Language 

# def call_llm(question:str):
#     llm  = ChatOpenAI(model="gpt-4o-mini")
#     llm_response = llm.invoke(question)
#     return llm, llm_response


# llm, llm_response = call_llm("Confirm if american soldiers are in Nigeria")
# print(llm_response, "\n\n")



# def parsing_output(llm_response):
#     from langchain_core.output_parsers import StrOutputParser
#     output_parser = StrOutputParser()
#     return output_parser, output_parser.invoke(llm_response)
    


# parser, parser_response = parsing_output(llm_response=llm_response)
# print(parser_response, "PARSER", "\n\n")


# def simple_chain(llm, output_parser, question):
#     chain = llm | output_parser
#     return chain.invoke(question)


# print(simple_chain(llm=llm, output_parser=parser, question="Tell me a Joke about Donald Trump" ))





# def structured_output(llm, startup_detail):
#     structured_llm = llm.with_structured_output(StartupReview)
#     return structured_llm.invoke(startup_detail)
  


# startup_details = """
# Paystack is a fintech company, that helps process payment both local and international businesses and individuals
# Paystack raised approximately $8 million in total funding before being acquired by Stripe in 2020.
# The company was founded by Shola Akinlade and Ezra Olubi.
# """


# structured_response = structured_output(llm=llm, startup_detail=startup_details)

# print("STRUCTURED","\n\n", structured_response)






# # Templates


# def chat_template(llm, parser, user_topic):
#     from langchain_core.prompts import ChatPromptTemplate
#     prompt = ChatPromptTemplate.from_template("Tell me more about {topic}", role="human")
#     chain = prompt | llm |parser
#     result = chain.invoke({"topic": user_topic})
#     return result


# chat_response = chat_template(llm=llm, parser=parser, user_topic="EIPSEIN FILES")
# print(chat_response, "EIPSEIN FILES")


# def llm_messages(llm):
#     from langchain_core.messages import HumanMessage, SystemMessage
#     system_message = SystemMessage(content="You are  a helpful assistant that tells jokes about world leaders")
#     human_message = HumanMessage(content="Tell me about Muhammed Gaddafi")
#     return llm.invoke([system_message, human_message])


# llm_messaging = llm_messages(llm=llm)
    

#Load multiple docs (allows pdf and docx) and split them into chunks for embedding and vector store creation
# def load_document(folder_path:str):
#     documents = []

#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#         elif filename.endswith(".docx"):
#             loader = Docx2txtLoader(file_path)
        
#         else:
#             print(f"Unsupported file type, {filename}")
#             continue

#         documents.extend(loader.load())
#         return documents

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# Text splitting and loading section

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)


pdf_loader = PyPDFLoader("comparison.pdf")
documents = pdf_loader.load()


# Load .docx file instead

# docx_loader = Docx2txtLoader("comparison.pdf")
# documents = docx_loader.load()


splits = text_splitter.split_documents(documents)
print(splits[0].page_content)
print(f"splitted documents into {len(splits)} chunks")



# Embed splitted contents into embeddins 
#1. use OpenAIEmbeddings
# embeddings = OpenAIEmbeddings()
# document_embedding = embeddings.embed_documents([split.page_content for split in splits])
# print(f"Create embeddings for {len(document_embedding)} document chunks")

# print(document_embedding[0])


#use self controlled lanchain transformers

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embedding_func =  SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
document_embeddings = embedding_func.embed_documents([split.page_content for split in splits])

from langchain_chroma import Chroma
collection_name = "my_collection"
vector_store = Chroma.from_documents(collection_name=collection_name, documents=splits, embedding=embedding_func, persist_directory="./chroma_db")

print("Vector db created")



#Similarity    search  and use the vector store as a retreiver


retriever = vector_store.as_retriever(search_kwargs = {"k":2})
retriever_response =retriever.invoke("What does fidex do ? ")


from langchain_core.prompts import ChatPromptTemplate
template = """

Answer the questions based on the following context and you are to return answers without mark downs so as reduce the stress of formatting for the user:

{context}

Question: {question}

Answer: 

"""

prompt = ChatPromptTemplate.from_template(template)
#Runnables

from langchain_core.runnables import RunnablePassthrough


def format_document_to_string(docs):
    # This function is used to format the documents from the retriever in a way that we just need  the page_content of the document retrieval
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain  =(
    {"context": retriever | format_document_to_string, "question": RunnablePassthrough()} | prompt
)

rag_chain.invoke("how does fidex compare or relates to Nubank ?")


#Lets now make it a full RAG + llm + parser for poper augmentaion and formatting 
from langchain_core.output_parsers import StrOutputParser

llm  = ChatOpenAI(model="gpt-4o-mini")
complete_rag_chain = (
    {"context": retriever | format_document_to_string, "question": RunnablePassthrough()}
    | prompt
    |llm
    |StrOutputParser() #parser

)

complete_rag_chain_question ="how does fidex compare or relates to Nubank ?"
complete_rag_chain_response = complete_rag_chain.invoke(complete_rag_chain_question)
print(complete_rag_chain_response, "CRR")
 


# Make the RAG conversational and allow follow up questions 
from langchain_core.messages import HumanMessage,AIMessage
chat_history = []
chat_history.extend([
    HumanMessage(content=complete_rag_chain_question),
    AIMessage(content=complete_rag_chain_response)
])




from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    """
        Given a chat history and the latest user question, which might reference in the chat history.
        Formulate a standalone question which can be understood.
         
        Without the chat history, DO NOT answer the question, just reformulate it if needed and otherwise return it as it is.

    """
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

contextualized_chain = contextualize_q_prompt | llm | StrOutputParser()

final_contextualized_res =contextualized_chain.invoke({"input": "Do they have any similarities", "chat_history":chat_history})


from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
    
)


history_aware_retriever_response =history_aware_retriever.invoke({"input": "Do they have any similarities", "chat_history":chat_history})
print("\n HISTORY RETRIEVER RESPONSE",history_aware_retriever_response)


from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful AI assistant. Use the following context to answer the user's question"),
        ("system","Context: {context}"),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human", "{input}")
        
    ]
)

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt, output_parser=StrOutputParser())


final_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
final_rag_response = final_rag_chain.invoke({"input": "Do they have any similarities", "chat_history":chat_history})
print("FINAL RAG RESPONSE ", "\n",final_rag_response)