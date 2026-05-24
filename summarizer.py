from transformers import pipeline
from .config import MODEL_NAME

summarizer = pipeline("summarization", model=MODEL_NAME)

def generate_summary(text: str):
    text = text[:4000]  # Avoid model token limits
    return summarizer(
        text, max_length=150, min_length=40, do_sample=False
    )[0]["summary_text"]

from transformers import MarianMTModel, MarianTokenizer

def get_translator(src, tgt):
    model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(text, src='fr', tgt='en'):
    tokenizer, model = get_translator(src, tgt)
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translation = model.generate(**tokens)
    return tokenizer.decode(translation[0], skip_special_tokens=True)

