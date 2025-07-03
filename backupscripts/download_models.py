# download_models.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import os

LOCAL_MODEL_DIR = 'D:/AI/Models/huggingface'
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

distilbert_path = os.path.join(LOCAL_MODEL_DIR, 'distilbert')
if not os.path.exists(distilbert_path):
    print(f"Downloading DistilBERT to {distilbert_path}...")
    model_name_distilbert = 'distilbert-base-uncased' 
    model = AutoModelForSequenceClassification.from_pretrained(model_name_distilbert)
    tokenizer = AutoTokenizer.from_pretrained(model_name_distilbert)
    model.save_pretrained(distilbert_path)
    tokenizer.save_pretrained(distilbert_path)
    print("DistilBERT downloaded and saved.")
else:
    print(f"DistilBERT already exists at {distilbert_path}.")

t5_path = os.path.join(LOCAL_MODEL_DIR, 't5')
if not os.path.exists(t5_path):
    print(f"Downloading T5 to {t5_path}...")
    model_name_t5 = 't5-small'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_t5)
    tokenizer = AutoTokenizer.from_pretrained(model_name_t5)
    model.save_pretrained(t5_path)
    tokenizer.save_pretrained(t5_path)
    print("T5 downloaded and saved.")
else:
    print(f"T5 already exists at {t5_path}.")

print("\nModel download process complete.")