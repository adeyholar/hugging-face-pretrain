# document_agent.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
# Import PDFGenerator from its separate script
from pdf_generator import PDFGenerator 

class DocumentAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 

        # Load quantized DistilBERT for sentiment analysis
        self.classifier = pipeline(
            "sentiment-analysis",
            model="D:/AI/Models/huggingface/distilbert_quantized",
            tokenizer="D:/AI/Models/huggingface/distilbert", 
        )
        
        # Load quantized T5 for summarization
        self.summarizer = pipeline(
            "summarization",
            model="D:/AI/Models/huggingface/t5_quantized",
            tokenizer="D:/AI/Models/huggingface/t5", 
        )

    def analyze_document(self, text: str) -> dict: # Added type hints
        # Sentiment analysis
        sentiment_results = self.classifier(text) 
        sentiment = sentiment_results[0] # type: ignore [reportOptionalSubscript, reportIndexIssue] 
        
        # Summarization
        # Adjust max_length and min_length as needed for your summaries
        summary_results = self.summarizer(text, max_length=150, min_length=40, do_sample=False)
        summary = summary_results[0]["summary_text"] # type: ignore [reportOptionalSubscript, reportIndexIssue, reportArgumentType]
        
        return {
            "sentiment": sentiment["label"], # type: ignore [reportArgumentType]
            "confidence": sentiment["score"], # type: ignore [reportArgumentType]
            "summary": summary
        }

    # Method to analyze and generate a PDF report
    def generate_report(self, text: str, output_dir: str = "reports") -> str: # Added return type hint
        analysis_result = self.analyze_document(text)
        pdf = PDFGenerator() # No arguments in init now
        
        # Call build method and capture the returned PDF path
        pdf_report_path = pdf.build(original_text=text, analysis_result=analysis_result, output_dir=output_dir)
        print(f"Report for '{text[:50]}...' saved to '{output_dir}'.")
        return pdf_report_path # Return the path for use in app.py


if __name__ == "__main__":
    agent = DocumentAgent()

    # --- Sample 1 ---
    sample_text_1 = "The product was amazing and exceeded expectations. The delivery was fast, and customer service was responsive. This is truly a revolutionary device that will change the industry."
    print("Analyzing sample text 1:")
    print(f"Original Text: {sample_text_1}")
    
    generated_path_1 = agent.generate_report(sample_text_1, output_dir="local_analysis_reports") 
    print(f"Report generated at: {generated_path_1}")

    # --- Sample 2 ---
    sample_text_2 = "This movie was absolutely terrible. The plot made no sense, and the acting was wooden. A complete waste of time and money. I would strongly advise against watching it, as it offers nothing redeeming."
    print("\nAnalyzing another sample text:")
    print(f"Original Text: {sample_text_2}")
    
    generated_path_2 = agent.generate_report(sample_text_2, output_dir="local_analysis_reports")
    print(f"Report generated at: {generated_path_2}")

    print("\nAll document analyses and local PDF reports completed.")