# document_agent.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
from pdf_generator import PDFGenerator 

# Define a base directory for models *inside the Docker container*
MODEL_BASE_DIR = os.getenv("MODEL_BASE_DIR", "/app/models") # Default to /app/models inside container

class DocumentAgent:
    def __init__(self):
        # Paths for the sentiment model
        sentiment_quantized_path = os.path.join(MODEL_BASE_DIR, 'sentiment_quantized')
        sentiment_tokenizer_path = os.path.join(MODEL_BASE_DIR, 'sentiment_model') # Tokenizer from original download path

        # Paths for the summarizer model
        t5_quantized_path = os.path.join(MODEL_BASE_DIR, 't5_quantized')
        t5_tokenizer_path = os.path.join(MODEL_BASE_DIR, 't5') # Tokenizer from original download path

        print(f"Loading sentiment classifier from: {sentiment_quantized_path}")
        if not os.path.exists(sentiment_quantized_path) or not os.path.exists(sentiment_tokenizer_path):
            print(f"ERROR: Sentiment model or tokenizer path missing! Quantized: {sentiment_quantized_path}, Tokenizer: {sentiment_tokenizer_path}")
            raise FileNotFoundError("Sentiment analysis model or tokenizer files not found in container.")

        self.classifier = pipeline(
            "sentiment-analysis",
            model=sentiment_quantized_path,
            tokenizer=sentiment_tokenizer_path,
        )
        
        print(f"Loading summarizer from: {t5_quantized_path}")
        if not os.path.exists(t5_quantized_path) or not os.path.exists(t5_tokenizer_path):
            print(f"ERROR: Summarizer model or tokenizer path missing! Quantized: {t5_quantized_path}, Tokenizer: {t5_tokenizer_path}")
            raise FileNotFoundError("Summarization model or tokenizer files not found in container.")

        self.summarizer = pipeline(
            "summarization",
            model=t5_quantized_path,
            tokenizer=t5_tokenizer_path, 
        )

        # Define the label mapping for the new sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest)
        self.sentiment_label_map = {
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'NEUTRAL',
            'LABEL_2': 'POSITIVE'
        }


    def analyze_document(self, text: str) -> dict:
        sentiment_results = self.classifier(text) 
        
        # --- DEBUG PRINT: Inspect raw sentiment results ---
        print(f"DEBUG: Raw sentiment results for '{text[:50]}...': {sentiment_results}")
        # ----------------------------------------------------

        sentiment = sentiment_results[0] # type: ignore [reportOptionalSubscript, reportIndexIssue, reportIncompatibleVariableType]
        
        # Map the generic label to a more descriptive one
        mapped_label = self.sentiment_label_map.get(sentiment["label"], sentiment["label"]) # Default to original if not found
        
        summary_results = self.summarizer(text, max_length=150, min_length=40, do_sample=False)
        
        # --- DEBUG PRINT: Inspect raw summary results ---
        print(f"DEBUG: Raw summary results for '{text[:50]}...': {summary_results}")
        # ----------------------------------------------------

        summary = summary_results[0]["summary_text"] # type: ignore [reportOptionalSubscript, reportIndexIssue, reportArgumentType, reportIncompatibleVariableType]
        
        return {
            "sentiment": mapped_label, # Use the mapped label here
            "confidence": sentiment["score"], # type: ignore [reportArgumentType]
            "summary": summary
        }

    def generate_report(self, text: str, output_dir: str = "reports") -> str:
        analysis_result = self.analyze_document(text)
        pdf = PDFGenerator()
        
        pdf_report_path = pdf.build(original_text=text, analysis_result=analysis_result, output_dir=output_dir)
        print(f"Report for '{text[:50]}...' saved to '{output_dir}'.")
        return pdf_report_path


if __name__ == "__main__":
    print("Running document_agent.py as standalone. Ensure models are accessible via MODEL_BASE_DIR env var or default path.")
    try:
        agent = DocumentAgent()

        sample_text_1 = "The product was amazing and exceeded expectations. The delivery was fast, and customer service was responsive. This is truly a revolutionary device that will change the industry."
        print("Analyzing sample text 1:")
        print(f"Original Text: {sample_text_1}")
        
        generated_path_1 = agent.generate_report(sample_text_1, output_dir="local_analysis_reports") 
        print(f"Report generated at: {generated_path_1}")

        sample_text_2 = "This movie was absolutely terrible. The plot made no sense, and the acting was wooden. A complete waste of time and money. I would strongly advise against watching it, as it offers nothing redeeming."
        print("\nAnalyzing another sample text:")
        print(f"Original Text: {sample_text_2}")
        
        generated_path_2 = agent.generate_report(sample_text_2, output_dir="local_analysis_reports")
        print(f"Report generated at: {generated_path_2}")

        print("\nAll document analyses and local PDF reports completed.")
    except Exception as e:
        print(f"Error during standalone document_agent run: {e}")
        print("Please ensure your models are correctly set up and accessible via the MODEL_BASE_DIR environment variable or the default path.")