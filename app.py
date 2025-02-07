from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os  # Add this

app = Flask(__name__)
CORS(app)

# SECURE API KEY USAGE (NEVER HARDCODE)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Store in environment variables


@app.route('/summarize', methods=['POST'])
def summarize():
    # ADD INPUT VALIDATION
    if not request.json or 'content' not in request.json:
        return jsonify({"error": "Missing content"}), 400

    data = request.json
    content = data['content'][:15000]  # Limit input size

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Summarize this document in 3-5 bullet points:"
            }, {
                "role": "user",
                "content": content
            }],
            temperature=0.3  # Add for more focused summaries
        )
        return jsonify({"summary": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    # ADD VALIDATION
    required_fields = ['context', 'history']
    if not all(field in request.json for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    data = request.json
    context = data['context'][:5000]  # Truncate document context

    # IMPROVED MESSAGE HANDLING
    messages = [
        {"role": "system", "content": f"Answer questions about: {context}"},
        *[{"role": msg["role"], "content": msg["content"][:1000]} for msg in data['history']]
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500  # Prevent long responses
        )
        return jsonify({
            "response": response.choices[0].message.content,
            "usage": response.usage  # Return token usage
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Disable debug in production
