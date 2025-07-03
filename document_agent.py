from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os

class DocumentAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 

        # Load quantized DistilBERT for sentiment analysis
        self.classifier = pipeline(
            "sentiment-analysis",
            model="D:/AI/Models/huggingface/distilbert_quantized",
            tokenizer="D:/AI/Models/huggingface/distilbert", 
            # REMOVED: device=0 if torch.cuda.is_available() else -1
        )
        
        # Load quantized T5 for summarization
        self.summarizer = pipeline(
            "summarization",
            model="D:/AI/Models/huggingface/t5_quantized",
            tokenizer="D:/AI/Models/huggingface/t5", 
            # REMOVED: device=0 if torch.cuda.is_available() else -1
        )

    def analyze_document(self, text):
        # Sentiment analysis
        sentiment_results = self.classifier(text) 
        sentiment = sentiment_results[0] # type: ignore [reportOptionalSubscript, reportIndexIssue] 
        
        # Summarization
        summary_results = self.summarizer(text, max_length=100, min_length=30, do_sample=False)
        summary = summary_results[0]["summary_text"] # type: ignore [reportOptionalSubscript, reportIndexIssue, reportArgumentType]
        
        return {
            # Add type: ignore for these specific Pylance warnings
            "sentiment": sentiment["label"], # type: ignore [reportArgumentType]
            "confidence": sentiment["score"], # type: ignore [reportArgumentType]
            "summary": summary
        }

if __name__ == "__main__":
    agent = DocumentAgent()
    sample_text = "The product was amazing and exceeded expectations. The delivery was fast, and customer service was responsive."
    print("Analyzing sample text:")
    print(f"Original Text: {sample_text}")
    result = agent.analyze_document(sample_text)
    print(f"Analysis Result: {result}")

    another_sample = "This movie was absolutely terrible. The plot made no sense, and the acting was wooden. A complete waste of time."
    print("\nAnalyzing another sample text:")
    print(f"Original Text: {another_sample}")
    result_negative = agent.analyze_document(another_sample)
    print(f"Analysis Result: {result_negative}")