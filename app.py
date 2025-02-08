from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import tempfile
import mammoth
import PyPDF2
import pandas as pd
from pptx import Presentation

app = Flask(__name__)
CORS(app)

# SECURE API KEY USAGE
openai.api_key = os.getenv("OPENAI_API_KEY")  # Store in environment variables


def extract_text_from_file(file_path, file_type):
    """Extracts text from different file formats."""
    try:
        if file_type == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()

        elif file_type == "docx":
            with open(file_path, "rb") as f:
                raw_text = mammoth.extract_raw_text(f)
            return raw_text.value.strip()

        elif file_type == "pptx":
            presentation = Presentation(file_path)
            text = "\n".join([shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text")])
            return text.strip()

        elif file_type == "xlsx":
            df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
            text = "\n".join([df[sheet].to_csv(index=False) for sheet in df])
            return text.strip()

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    except Exception as e:
        raise ValueError(f"Failed to extract text from {file_type}: {str(e)}")


@app.route('/summarize', methods=['POST'])
def summarize():
    """Receives a file upload, extracts text, and sends it to OpenAI API."""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded_file = request.files['file']
        file_name = uploaded_file.filename
        file_type = os.path.splitext(file_name)[-1].lower().replace('.', '')

        # Validate file type
        allowed_types = {"pdf", "docx", "pptx", "xlsx"}
        if file_type not in allowed_types:
            return jsonify({"error": f"Unsupported file type: {file_type}"}), 400

        # Save file to a temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
        uploaded_file.save(temp_file_path)

        # Extract text from file
        extracted_text = extract_text_from_file(temp_file_path, file_type)

        # Ensure valid text extraction
        if not extracted_text.strip():
            return jsonify({"error": "No text extracted from file"}), 400

        # Send extracted content to OpenAI for summarization
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Summarize this document in 3-5 bullet points:"},
                      {"role": "user", "content": extracted_text[:15000]}],  # Limit input size
            temperature=0.3
        )

        return jsonify({"summary": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Disable debug in production
