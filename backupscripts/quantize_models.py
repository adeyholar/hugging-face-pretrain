# quantize_models.py

from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
# Pylance fix: Recommended import path
from transformers.utils.quantization_config import BitsAndBytesConfig 
import torch
import os

QUANTIZED_MODEL_DIR = 'D:/AI/Models/huggingface'
os.makedirs(QUANTIZED_MODEL_DIR, exist_ok=True)

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

print("Loading and quantizing DistilBERT...")
try:
    distilbert_original_path = os.path.join(QUANTIZED_MODEL_DIR, 'distilbert')
    distilbert_quantized_path = os.path.join(QUANTIZED_MODEL_DIR, 'distilbert_quantized')
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(
        distilbert_original_path,
        quantization_config=quantization_config
    )
    distilbert_model.save_pretrained(distilbert_quantized_path)
    distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_original_path)
    distilbert_tokenizer.save_pretrained(distilbert_quantized_path)
    print("DistilBERT quantized and saved (model and tokenizer).")
except Exception as e:
    print(f"Error quantizing DistilBERT: {e}")
    print("Ensure 'bitsandbytes' is installed and CUDA is available if trying to quantize on GPU.")


print("\nLoading and quantizing T5...")
try:
    t5_original_path = os.path.join(QUANTIZED_MODEL_DIR, 't5')
    t5_quantized_path = os.path.join(QUANTIZED_MODEL_DIR, 't5_quantized')
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        t5_original_path,
        quantization_config=quantization_config
    )
    t5_model.save_pretrained(t5_quantized_path)
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_original_path)
    t5_tokenizer.save_pretrained(t5_quantized_path)
    print("T5 quantized and saved (model and tokenizer).")
except Exception as e:
    print(f"Error quantizing T5: {e}")
    print("Ensure 'bitsandbytes' is installed and CUDA is available if trying to quantize on GPU.")

print("\nQuantization process complete.")