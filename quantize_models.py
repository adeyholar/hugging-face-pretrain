# quantize_models.py

from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig 
import torch
import os

# Define quantized models directory, matching MODEL_BASE_DIR
QUANTIZED_MODEL_DIR = os.getenv("MODEL_BASE_DIR", "/app/models")
os.makedirs(QUANTIZED_MODEL_DIR, exist_ok=True)

# Define quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# --- Quantize Sentiment Model ---
sentiment_original_path = os.path.join(QUANTIZED_MODEL_DIR, 'sentiment_model')
sentiment_quantized_path = os.path.join(QUANTIZED_MODEL_DIR, 'sentiment_quantized')

print("Loading and quantizing Sentiment Model...")
try:
    # Ensure this is AutoModelForSequenceClassification for the sentiment task
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        sentiment_original_path,
        quantization_config=quantization_config
    )
    sentiment_model.save_pretrained(sentiment_quantized_path)
    
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_original_path)
    sentiment_tokenizer.save_pretrained(sentiment_quantized_path)
    print("Sentiment model quantized and saved (model and tokenizer).")
except Exception as e:
    print(f"Error quantizing Sentiment Model: {e}")
    print(f"Ensure original sentiment model is in {sentiment_original_path}")


# --- Quantize T5 ---
t5_original_path = os.path.join(QUANTIZED_MODEL_DIR, 't5')
t5_quantized_path = os.path.join(QUANTIZED_MODEL_DIR, 't5_quantized')

print("\nLoading and quantizing T5...")
try:
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
    print(f"Ensure original T5 model is in {t5_original_path}")

print("\nQuantization process complete.")