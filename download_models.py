# download_models.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Define the local directory to save models *within the Docker environment*
LOCAL_MODEL_DIR = os.getenv("MODEL_BASE_DIR", "/app/models") # Default to /app/models
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Hugging Face model names to download
# Use a model explicitly fine-tuned for sentiment analysis
HF_SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst2" # This is a common sentiment model
HF_T5_NAME = "t5-small"

# --- Download and Save Sentiment Model ---
sentiment_model_path = os.path.join(LOCAL_MODEL_DIR, 'sentiment_model') # Give it a clearer name
if not os.path.exists(sentiment_model_path):
    print(f"Downloading Sentiment Model '{HF_SENTIMENT_MODEL_NAME}' to {sentiment_model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(HF_SENTIMENT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_SENTIMENT_MODEL_NAME) # Use the same name for tokenizer
    
    model.save_pretrained(sentiment_model_path)
    tokenizer.save_pretrained(sentiment_model_path)
    print("Sentiment model downloaded and saved.")
else:
    print(f"Sentiment model already exists at {sentiment_model_path}.")

# --- Download and Save T5 for Summarization ---
t5_path = os.path.join(LOCAL_MODEL_DIR, 't5')
if not os.path.exists(t5_path):
    print(f"Downloading T5 '{HF_T5_NAME}' to {t5_path}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_T5_NAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_T5_NAME)
    
    model.save_pretrained(t5_path)
    tokenizer.save_pretrained(t5_path)
    print("T5 downloaded and saved.")
else:
    print(f"T5 already exists at {t5_path}.")

print("\nModel download process complete.")