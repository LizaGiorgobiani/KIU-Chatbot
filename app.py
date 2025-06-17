from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from main import ChatbotAssistant

app = Flask(__name__)
CORS(app)

# Initialize chatbot
assistant = ChatbotAssistant('intents.json')
assistant.parse_intents()
assistant.prepare_data()
assistant.load_model('chatbot_model.pth', 'dimensions.json')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    response = assistant.process_message(message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
