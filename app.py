import os
from huggingface_hub import InferenceClient
from flask import Flask, render_template, request
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    client = InferenceClient(
        provider="hf-inference",
        api_key=os.getenv("HF_TOKEN"),
    )

    question = request.form['question']

    result = client.text_classification(
        question,
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    return render_template('index.html', ans= result[0])

if __name__ == "__main__":
    app.run(port=5000)