# app.py

from flask import Flask, request, render_template_string, send_file, jsonify
from document_agent import DocumentAgent
import os

app = Flask(__name__)
agent = DocumentAgent()

# Define where the reports will be saved by the DocumentAgent
REPORTS_FOLDER = "analysis_reports"
os.makedirs(REPORTS_FOLDER, exist_ok=True) # Ensure this directory exists

# Basic HTML template for the upload form
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
        input[type="file"] {
            border: 1px solid #a7d9f7;
            padding: 10px;
            border-radius: 5px;
            width: 70%;
            margin-bottom: 20px;
            background-color: #f0f8ff;
            color: #333;
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
        .footer { text-align: center; margin-top: 30px; font-size: 0.9em; color: #777; }
        .error-message { color: red; font-weight: bold; margin-top: 10px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload Document for AI Analysis</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".txt"><br>
            <p style="font-size: 0.9em; color: #555;">Please upload a plain text (.txt) file for analysis.</p>
            <input type="submit" value="Analyze and Download Report">
        </form>
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
        # Check if a file was submitted
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files["file"]
        
        # Check if a file was selected
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if file:
            try:
                # Read the file content. Assumes UTF-8 for .txt files.
                text = file.read().decode("utf-8")
                
                # Use the DocumentAgent to analyze and generate the report.
                # The generate_report method now returns the full path of the generated PDF.
                pdf_report_path = agent.generate_report(text, output_dir=REPORTS_FOLDER)
                
                # Extract the base filename for the download
                download_name = os.path.basename(pdf_report_path)
                
                # Send the generated PDF file back to the client for download
                return send_file(
                    pdf_report_path, 
                    as_attachment=True, 
                    download_name=download_name, # This sets the filename for the user's download
                    mimetype='application/pdf' # Explicitly set mimetype
                )
            except UnicodeDecodeError:
                return jsonify({"error": "Failed to decode file. Please ensure it's a plain text (UTF-8) file."}), 400
            except Exception as e:
                # Catch any other general errors during processing
                return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
    
    # For GET requests, render the HTML form
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    # Run the Flask app in debug mode (useful for development)
    # host="0.0.0.0" makes it accessible from other devices on the network, not just localhost
    # port=5000 is the default Flask port
    app.run(debug=True, host="0.0.0.0", port=5000)