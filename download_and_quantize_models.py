from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils.quantization_config import BitsAndBytesConfig
import os
import torch

# Use environment variables for model names
DISTILBERT_MODEL_NAME = os.getenv("DISTILBERT_MODEL_NAME", "distilbert-base-uncased")
T5_MODEL_NAME = os.getenv("T5_MODEL_NAME", "t5-small")
MODEL_DIR = "/app/models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Define quantization configuration
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Download and quantize DistilBERT
print(f"Downloading and quantizing DistilBERT ({DISTILBERT_MODEL_NAME})...")
distilbert_path = os.path.join(MODEL_DIR, "distilbert")
if not os.path.exists(distilbert_path):
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
    model.save_pretrained(distilbert_path)
    tokenizer.save_pretrained(distilbert_path)
distilbert_quantized_path = os.path.join(MODEL_DIR, "distilbert_quantized")
model = AutoModelForSequenceClassification.from_pretrained(distilbert_path, quantization_config=quantization_config)
model.save_pretrained(distilbert_quantized_path)
tokenizer.save_pretrained(distilbert_quantized_path)
print("DistilBERT downloaded and quantized.")

# Download and quantize T5
print(f"Downloading and quantizing T5 ({T5_MODEL_NAME})...")
t5_path = os.path.join(MODEL_DIR, "t5")
if not os.path.exists(t5_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    model.save_pretrained(t5_path)
    tokenizer.save_pretrained(t5_path)
t5_quantized_path = os.path.join(MODEL_DIR, "t5_quantized")
model = AutoModelForSeq2SeqLM.from_pretrained(t5_path, quantization_config=quantization_config)
model.save_pretrained(t5_quantized_path)
tokenizer.save_pretrained(t5_quantized_path)
print("T5 downloaded and quantized.")

print("\nModel download and quantization process complete.")