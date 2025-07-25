# app.py

from flask import Flask, request, render_template_string, send_file, jsonify
from document_agent import DocumentAgent
from email_service import EmailService 
import os
import secrets 

app = Flask(__name__)

# Initialize your document analysis agent
# This will trigger model loading at app startup
agent = DocumentAgent() 

# Initialize your email service
email_service = EmailService() 

# Define where the reports will be saved by the DocumentAgent
REPORTS_FOLDER = "analysis_reports"
os.makedirs(REPORTS_FOLDER, exist_ok=True) # Ensure this directory exists

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Document Analysis</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #e9f2f9; color: #333; line-height: 1.6; }
        .container { max-width: 800px; margin: auto; background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        h1 { color: #0056b3; text-align: center; margin-bottom: 30px; }
        form { display: flex; flex-direction: column; align-items: center; }
        input[type="file"], input[type="email"], input[type="text"] {
            border: 1px solid #a7d9f7;
            padding: 10px;
            border-radius: 5px;
            width: 70%;
            margin-bottom: 20px;
            background-color: #f0f8ff;
            color: #333;
            box-sizing: border-box;
        }
        input[type="submit"] {
            background-color: #007bff;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 17px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        input[type="submit"]:hover { background-color: #0056b3; }
        .message {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }
        .footer { text-align: center; margin-top: 30px; font-size: 0.9em; color: #777; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload Document for AI Analysis</h1>
        <form method="post" enctype="multipart/form-data">
            <label for="file">Select Document (.txt):</label>
            <input type="file" name="file" id="file" accept=".txt"><br>
            
            <label for="to_email">Recipient Email (Optional):</label>
            <input type="email" name="to_email" id="to_email" placeholder="client@example.com"><br>

            <label for="email_subject">Email Subject (Optional):</label>
            <input type="text" name="email_subject" id="email_subject" placeholder="Your Report Subject"><br>
            
            <input type="submit" value="Analyze and Get Report">
        </form>
        {% if message %}
            <div class="message {{ message_type }}">{{ message }}</div>
        {% endif %}
    </div>
    <div class="footer">
        <p>Powered by Hugging Face Transformers & ReportLab</p>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template_string(HTML_TEMPLATE, message="No file part in the request.", message_type="error")
        
        file = request.files["file"]
        to_email = request.form.get("to_email")
        email_subject = request.form.get("email_subject") or "Your Document Analysis Report"
        
        if file.filename == "":
            return render_template_string(HTML_TEMPLATE, message="No file selected.", message_type="error")
        
        if file:
            try:
                text = file.read().decode("utf-8")
                pdf_report_path = agent.generate_report(text, output_dir=REPORTS_FOLDER)
                
                email_message = "Report generated successfully. "
                message_type = "success"

                if to_email:
                    email_body = (
                        "Dear client,\n\n"
                        "Please find your AI document analysis report attached. "
                        "The report provides sentiment analysis and a summary of your uploaded document.\n\n"
                        "Best regards,\n"
                        "Your AI Analysis Service"
                    )
                    try:
                        email_service.send_email(
                            to_email=to_email,
                            subject=email_subject,
                            body=email_body,
                            attachment_path=pdf_report_path
                        )
                        email_message += f"Email sent to {to_email}."
                        message_type = "success"
                    except ValueError as ve:
                        email_message += f"Email configuration error: {ve}. Email not sent."
                        message_type = "error"
                    except Exception as email_err:
                        email_message += f"Failed to send email to {to_email}: {email_err}. "
                        message_type = "error"
                else:
                    email_message += "No email address provided for delivery."
                    message_type = "info"
                
                print(f"Web interface outcome: {email_message}") # For server logs
                
                download_name = os.path.basename(pdf_report_path)
                return send_file(
                    pdf_report_path, 
                    as_attachment=True, 
                    download_name=download_name, 
                    mimetype='application/pdf'
                )

            except UnicodeDecodeError:
                return render_template_string(HTML_TEMPLATE, message="Failed to decode file. Please ensure it's a plain text (UTF-8) file.", message_type="error")
            except Exception as e:
                return render_template_string(HTML_TEMPLATE, message=f"An internal server error occurred during analysis: {str(e)}", message_type="error")
    
    return render_template_string(HTML_TEMPLATE, message=None)

if __name__ == "__main__":
    # For production deployment with Docker, it's better to use a WSGI server like Gunicorn
    # For simple local testing with Docker, Flask's built-in server can suffice.
    # debug=False and threaded=True for better production characteristics
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)