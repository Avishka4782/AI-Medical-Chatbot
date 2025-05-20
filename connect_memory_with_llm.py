import os
os.environ["TOGETHER_API_KEY"] = "tgp_v1_-ARD5dhs6CcRriLssyvLiVKvaZzj76wKCFr6pb-W_Fc"

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Together



## Uncomment the following files if you're not using pipenv as your virtual environment manager
DB_FAISS_PATH = "vectorstor/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_FAISS_PATH, 
                               embeddings=embedding_model, 
                               index_name="index",
                               allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()


# Step 1: Setup LLM (Mistral with HuggingFace)


llms = Together(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.7,
    max_tokens=512
)
# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=llms,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
