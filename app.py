import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from groq import Groq

def format_answer(raw_text):
    """
    Formats AI output for better readability in HTML.
    Splits sections and adds bullets or numbered lists.
    """
    if not raw_text:
        return ""
    
    html = raw_text.replace("\n\n", "<br><br>").replace("- ", "• ").replace("1.", "<br>1.").replace("2.", "<br>2.")
    return html

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For embeddings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")      # For chat

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
client = Groq(api_key=GROQ_API_KEY)

persist_dir = "chroma_db"
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
print("✅ Loaded vectorstore from disk.")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    chunks = []
    query = ""

    if request.method == "POST":
        query = request.form["query"]

        # Retrieve top 5 chunks (increase k if needed)
        results = vectorstore.similarity_search(query, k=5)
        chunks = [doc.page_content[:500] + "..." for doc in results]

        # Prepare prompt for Groq GPT
        context = "\n\n".join([doc.page_content for doc in results])
        prompt = f"""
You are an intelligent AI assistant. 
Use ONLY the following context to answer the question clearly and accurately.
If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer:
"""

        # Generate answer using Groq GPT
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b"
        )
        raw_answer = response.choices[0].message.content
        answer = format_answer(raw_answer)

    return render_template("index.html", query=query, chunks=chunks, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)

