from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os

app = Flask(__name__)
CORS(app)

# Example PDF file path (replace with your actual file path)
pdf_path = "Document2.pdf"

# extract the text
with open(pdf_path, "rb") as pdf_file:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

# split into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text)

# Retrieve OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
knowledge_base = FAISS.from_texts(chunks, embeddings)

@app.route('/ask', methods=['POST'])
def ask_question():
    if request.method == 'OPTIONS':
        return '', 204  # No content, indicating success for preflight request

    try:
        # Get question from the frontend
        user_question = request.json['question']

        # ask a question
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI(openai_api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)

        return jsonify({'response': response, 'callback': str(cb)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the app without ngrok
    app.run()
