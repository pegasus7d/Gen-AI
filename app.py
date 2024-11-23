from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import tempfile
import os
from transformers import pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import requests

app = Flask(__name__)
CORS(app)

# Initialize the Whisper model for transcription
model = WhisperModel("base", device="cpu")

# Initialize Haystack for RAG (Retrieval-Augmented Generation)
document_store = InMemoryDocumentStore()
retriever = DensePassageRetriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
qa_pipeline = ExtractiveQAPipeline(reader, retriever)

# Initialize NLP pipeline for question identification
question_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# SERP API Key (replace with your actual key)
SERP_API_KEY = "c0ac07e704d8dcc6a4d8dba61aefb19acfd0d748948a73696e24800db081891e"

# Prepopulate the document store with documents
def populate_document_store():
    """Populate the document store with predefined documents."""
    if document_store.get_document_count() == 0:
        print("Document store is empty. Populating with predefined documents...")
        documents = [
            {"content": "Blockchain is a distributed ledger technology used for secure and transparent transactions."},
            {"content": "Artificial Intelligence is the simulation of human intelligence in machines."},
            {"content": "Python is a versatile programming language used for web development, data analysis, and machine learning."},
            {"content": "Cloud computing allows on-demand availability of computing resources over the internet."},
            {"content": "Quantum computing uses quantum mechanics to perform complex calculations much faster than traditional computers."}
        ]
        document_store.write_documents(documents)
        print(f"Added {len(documents)} documents to the document store.")
        print("Updating embeddings for the document store...")
        document_store.update_embeddings(retriever)
        print(f"Added {len(documents)} documents to the document store with embeddings.")
    else:
        print("Document store already contains documents.")

# Populate the document store at application startup
populate_document_store()

def transcribe_audio(audio_file):
    """Transcribe audio using Faster Whisper."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        audio_file.save(temp_file.name)
        temp_audio_path = temp_file.name
    segments, _ = model.transcribe(temp_audio_path)
    transcription = " ".join([segment.text for segment in segments])
    os.remove(temp_audio_path)
    return transcription


def identify_question(transcription):
    """Identify if there's a question in the transcription."""
    result = question_classifier(transcription, candidate_labels=["question", "statement"])
    is_question = result["labels"][0] == "question"
    return is_question, transcription


def decide_route(question):
    """Decide whether to use RAG, web search, or direct LLM response."""
    web_search_keywords = ["what", "when", "where", "who", "why", "how", "current", "latest"]
    
    # Check if the question contains any keyword for web search
    if any(keyword in question.lower() for keyword in web_search_keywords):
        return "web_search"
    elif document_store.get_document_count() > 0:
        return "rag"
    else:
        return "llm"


def handle_rag(question):
    """Handle RAG-based retrieval and answer generation."""
    try:
        results = qa_pipeline.run(query=question)
        answers = results.get("answers", [])
        
        if answers:
            # Access the first answer's 'answer' attribute
            return answers[0].answer
        else:
            return "No relevant answers found in the document store."
    except Exception as e:
        return f"Error during RAG processing: {str(e)}"


def handle_llm(question):
    """Handle answering directly with an LLM."""
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")  # You can use a GPT-based model
    return generator(question, max_length=150, num_return_sequences=1, temperature=0.7, top_p=0.9, repetition_penalty=1.2)[0]["generated_text"]


def handle_web_search(question):
    """Perform a web search using SERP API and return the results."""
    try:
        url = "https://serpapi.com/search"
        params = {
            "q": question,
            "api_key": SERP_API_KEY,
            "num": 3  # Number of results to retrieve
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            search_results = response.json()
            organic_results = search_results.get("organic_results", [])
            # Format the results into a readable answer
            answer = "\n".join(
                [f"{i+1}. {result.get('title', 'No Title')}: {result.get('snippet', 'No Snippet')} (URL: {result.get('link', 'No Link')})"
                 for i, result in enumerate(organic_results)]
            )
            return answer or "No relevant results found."
        else:
            return f"Web search failed with status code {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error during web search: {str(e)}"


@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process uploaded audio and return transcription, question, decision, and answer."""
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        # Transcribe the audio
        transcription = transcribe_audio(audio_file)

        # Identify if there's a question
        is_question, question_or_statement = identify_question(transcription)

        if not is_question:
            # If no question is detected, use RAG as a fallback
            if document_store.get_document_count() > 0:
                decision = "rag"
                answer = handle_rag(transcription)  # Use the transcription as the query for RAG
            else:
                decision = "N/A"
                answer = "No question detected and no documents available for retrieval."

            # Log and return the result
            print({
                "transcription": transcription,
                "question": "No question detected.",
                "decision": decision,
                "answer": answer
            })

            return jsonify({
                "transcription": transcription,
                "question": "No question detected.",
                "decision": decision,
                "answer": answer
            })

        # Decide response route (RAG, web search, or LLM)
        decision = decide_route(question_or_statement)

        # Generate answer based on the decision
        if decision == "rag":
            answer = handle_rag(question_or_statement)
        elif decision == "web_search":
            answer = handle_web_search(question_or_statement)
        else:
            answer = handle_llm(question_or_statement)

        # Log the results
        print({
            "transcription": transcription,
            "question": question_or_statement,
            "decision": decision,
            "answer": answer
        })

        # Return the results
        return jsonify({
            "transcription": transcription,
            "question": question_or_statement,
            "decision": decision,
            "answer": answer
        })

    except Exception as e:
        # Log the error to the console
        print({"error": str(e)})
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
