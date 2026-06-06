import os
import json
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
DATA_FILE = "users.json"

# Helper function to read users from the local disk file
def read_users_from_disk():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

# Helper function to write users safely back to the local disk
def write_users_to_disk(users_list):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(users_list, f, indent=4)
        return True
    except Exception:
        return False

@app.route('/')
def index():
    # Serves the auth.html file
    try:
        with open("auth.html", "r", encoding="utf-8") as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return "auth.html file not found on disk.", 404

# API Endpoint: Fetch all registered users
@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify(read_users_from_disk())

# API Endpoint: Save a user list or individual registration updates to disk
@app.route('/api/users', methods=['POST'])
def save_users():
    data = request.get_json()
    if not isinstance(data, list):
        return jsonify({"error": "Invalid data format. Expected a list of users."}), 400
    
    if write_users_to_disk(data):
        return jsonify({"status": "success", "message": "Data written to local disk successfully."})
    else:
        return jsonify({"error": "Failed to write data to local disk."}), 500

if __name__ == '__main__':
    # Starts the server on http://127.0.0.1:5000
    app.run(debug=True, port=5000)
