# email_service.py

from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import smtplib
from dotenv import load_dotenv
import os

class EmailService:
    def __init__(self):
        load_dotenv() # Load environment variables from .env file
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.username = os.getenv("SMTP_USERNAME")
        self.password = os.getenv("SMTP_PASSWORD")

    def send_email(self, to_email: str, subject: str, body: str, attachment_path: str):
        # This check ensures username and password are not None before use
        if not all([self.username, self.password]):
            raise ValueError("SMTP credentials (username or password) not configured in .env file.")
        
        msg = MIMEMultipart()
        # Pylance fix: Explicitly cast to str after the None check
        msg["From"] = str(self.username) 
        msg["To"] = to_email
        msg["Subject"] = subject
        
        msg.attach(MIMEText(body, "plain"))
        
        try:
            with open(attachment_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
                part["Content-Disposition"] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                msg.attach(part)
        except FileNotFoundError:
            print(f"Error: Attachment file not found at {attachment_path}. Sending email without attachment.")
            # Optionally, you could re-raise or handle this more robustly
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
                # Pylance fix: Explicitly cast to str after the None check
                server.login(str(self.username), str(self.password)) 
                server.send_message(msg)
            print(f"Email sent successfully to {to_email} with attachment {attachment_path}")
        except smtplib.SMTPAuthenticationError:
            print(f"SMTP Authentication Error: Check your username and app password for {self.username}. (For Gmail, use App Password)")
            raise
        except Exception as e:
            print(f"Failed to send email to {to_email}: {e}")
            raise

if __name__ == "__main__":
    # Example usage (replace with your credentials and test email)
    load_dotenv()
    
    # Create a dummy PDF for testing if it doesn't exist
    if not os.path.exists("sample_report.pdf"):
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        doc = SimpleDocTemplate("sample_report.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("This is a Sample PDF Report.", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("This file is for testing email attachments.", styles["BodyText"]))
        doc.build(story)
        print("Generated dummy sample_report.pdf for testing.")


    email_service = EmailService()
    test_to_email = os.getenv("TEST_EMAIL_RECIPIENT", "your_recipient_email@example.com") # Add TEST_EMAIL_RECIPIENT to .env or change
    try:
        email_service.send_email(
            to_email=test_to_email,
            subject="Test Document Analysis Report",
            body="Dear client,\n\nPlease find your document analysis report attached.\n\nBest regards,\nYour AI Agent",
            attachment_path="sample_report.pdf" # Make sure this file exists for the test
        )
    except ValueError as e:
        print(f"Configuration Error: {e}. Please check your .env file.")
    except Exception as e:
        print(f"An error occurred during email test: {e}")