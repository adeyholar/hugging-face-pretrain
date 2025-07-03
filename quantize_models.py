# THIS IS THE CORRECT IMPORT FOR TRANSFORMERS 4.53.0
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, BitsAndBytesConfig 
import torch
import os

# Ensure quantized models directory exists
os.makedirs('D:/AI/Models/huggingface', exist_ok=True)

# Define quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# Quantize DistilBERT
print("Loading and quantizing DistilBERT...")
distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    'D:/AI/Models/huggingface/distilbert', # Loading from your local path
    quantization_config=quantization_config
)
distilbert_model.save_pretrained('D:/AI/Models/huggingface/distilbert_quantized')
print("DistilBERT quantized and saved.")

# Quantize T5
print("\nLoading and quantizing T5...")
t5_model = AutoModelForSeq2SeqLM.from_pretrained(
    'D:/AI/Models/huggingface/t5', # Loading from your local path
    quantization_config=quantization_config
)
t5_model.save_pretrained('D:/AI/Models/huggingface/t5_quantized')
print("T5 quantized and saved.")

print("\nQuantization process complete.")