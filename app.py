import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from groq import Groq

def format_answer(raw_text):
    """
    Formats AI output for better readability in HTML.
    Handles paragraphs, bullets, and numbered lists.
    """
    if not raw_text:
        return ""

    lines = raw_text.split("\n")
    html_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
     
        if line.startswith("- "):
            html_lines.append(f"• {line[2:]}")

        elif len(line) > 2 and line[0:2].isdigit() and line[2] == ".":
            html_lines.append(line)
        else:
            html_lines.append(line)

    html = "\n".join(html_lines)
    return html

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")      

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
persist_dir = "chroma_db"
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
print("Loaded vectorstore from disk.")

client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    answer = None
    chunks = []

    if request.method == "POST":
        query = request.form.get("query", "").strip()

        if not query:
            answer = "Please enter a question."
        else:
            results = vectorstore.similarity_search(query, k=5)
            chunk_summaries = []
            for i, doc in enumerate(results, start=1):
                summary = doc.page_content[:300].replace("\n", " ")  # trim and remove newlines
                chunk_summaries.append(f"{i}. {summary}...")
            if not results:

                answer = "I couldn't find any relevant information in the documents."
            else:
                chunks = [
                    doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    for doc in results
                ]

                context = "\n\n".join([doc.page_content[:1000] for doc in results])  # Trim to avoid long context
                prompt = f"""
You are a helpful and knowledgeable AI assistant.
You have access to reference information. Use it only if it is relevant, and you may provide additional reasoning, explanations, examples, or step by step guidance.
Write in a friendly and engaging tone.
If the reference information does not fully answer the question, use your general knowledge to provide a helpful answer.
HTML is forbidden. Never use br or any other tag.
All output must be plain text only, with real newline characters.
Do not use formatting such as bold, italics, emojis, tables, or special characters.

Reference Information: {chunk_summaries}
Question: {query}

Answer:
If you use the reference information, you may mention the source chunk number.
"""

                try:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="openai/gpt-oss-120b"
                    )
                    raw_answer = response.choices[0].message.content
                    answer = format_answer(raw_answer)
                except Exception as e:
                    answer = f"Error generating answer: {e}"

    return render_template("index.html", query=query, chunks=chunks, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
