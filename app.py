import json
from flask import Flask
from flask import request
from flask.helpers import make_response
from flask.json import JSONDecoder, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/test', methods=['GET', 'POST'])
def hello_world():
    d = json.loads(request.data)
    return jsonify(
        d
        )
