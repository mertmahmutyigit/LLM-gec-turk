
from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re

app = Flask(__name__)

model_name = "Mert1315/Tr-grammer-mt5-base"
api_token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_token)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=api_token)

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences

@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    output_text = ""
    if request.method == "POST":
        input_text = request.form["input_text"]
        if request.form["action"] == "temizle":
            input_text = ""
            output_text = ""
        else:
            sentences = split_sentences(input_text)
            output_parts = []
            for sentence in sentences:
                inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_length=512,
                        num_beams=4,
                        early_stopping=True
                    )
                corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
                output_parts.append(corrected)
            output_text = " ".join(output_parts)
    return render_template("index.html", input_text=input_text, output_text=output_text)

if __name__ == "__main__":
    app.run(debug=True)
