# import os
# from dotenv import load_dotenv
# from google import genai


# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# client = genai.Client(api_key=GOOGLE_API_KEY)

# response = client.models.generate_content(
#     model="gemini-2.0-flash", contents="Explain how AI works in a few words"
# )
# print(response.text)

import os
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


#load the environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#initialize the Google Generative AI client
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)

#load the text file
loader = PyPDFLoader("dsa1.pdf", "dsa2.pdf")  # Change to TextLoader if you have a text file
docs = []
for pdf_file in ["dsa1.pdf", "dsa2.pdf"]:
    loader = PyPDFLoader(pdf_file)
    docs.extend(loader.load())

#split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

#initialize the vector store
vector_store = FAISS.from_documents(split_docs, embeddings)

#initialize the retrieval QA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)


#appointment booking system
def collect_user_info():
    print("\n📞 Let's collect your contact information.")
    name = input("Your Name: ")
    phone = input("Your Phone Number: ")
    email = input("Your Email: ")
    print("\nThank you! Our team will contact you soon.")
    # You can add logic here to store or process the collected info
    return {"name": name, "phone": phone, "email": email}



#ask a question
print("💬 Ask me anything about your document (type 'exit' to quit)\n")

while True:
    question = input("You: ")
    if question.lower() in ['exit', 'quit']:
        break
    
    # Check if user wants to be called
    if "call me" in question.lower() or "contact me" in question.lower():
        user_info = collect_user_info()
        print(f"Collected info: {user_info}\n")
        continue

    output = retrieval_qa.invoke({"query": question})
    print("Gemini:", output["result"], "\n")


