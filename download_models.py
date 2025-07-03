from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Ensure models directory exists
os.makedirs('D:/AI/Models/huggingface', exist_ok=True)

# DistilBERT for classification
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer.save_pretrained('D:/AI/Models/huggingface/distilbert')
model.save_pretrained('D:/AI/Models/huggingface/distilbert')

# T5 for summarization
t5_model = 't5-small'
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model)
t5_tokenizer.save_pretrained('D:/AI/Models/huggingface/t5')
t5_model.save_pretrained('D:/AI/Models/huggingface/t5')