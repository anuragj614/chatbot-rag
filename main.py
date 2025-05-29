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

import os, re, dateparser
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

def validate_email(email):
    """Validate the email format."""
    if not email:
        return False
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def validate_phone(phone):
    """Validate the phone number format."""
    if not phone:
        return False           
    return re.match(r"^\+?[1-9]\d{1,14}$", phone)

#ask a question
print("Ask me anything about your document (type 'exit' to quit)\n")

while True:
    question = input("You: ")
    if question.lower() in ['exit', 'quit']:
        break


    if "call me" in question.lower() or "book appointment" in question.lower():
        print("Okay! Let's book an appointment.")
    
    # Step 1: Ask for date
        date_query = input("When should we contact you? (e.g. on June 9th): ")
        parsed_date = dateparser.parse(date_query)
    
        if not parsed_date:
            print("Couldn't understand the date. Please try again.\n")
            continue
    
        # Step 2: Ask for name, phone, email
        name = input("Your name: ")
        
        while True:
            phone = input("Your phone number (format: +1234567890): ")
            if validate_phone(phone):
                break
            
            print("Invalid phone number format. Please try again.")
            
        while True:
            email = input("Your email address: ")
            if validate_email(email):
                break
            
            print("Invalid email format. Please try again.")    
    
        # Step 3: Format and display confirmation
        print("\n Appointment booked!")
        print(f"Name: {name}")
        print(f"Phone: {phone}")
        print(f"Email: {email}")
        print(f"Date: {parsed_date.strftime('%Y-%m-%d')}\n")
        continue


    # Default: document Q&A
    concise_query = f"Answer concisely: {question}"
    output = retrieval_qa.invoke({"query": concise_query})
    print("Gemini:", output["result"], "\n")


