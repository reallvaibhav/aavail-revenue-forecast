import os
from flask import Flask, jsonify, request, send_from_directory
from logger import LOG_DIR
from model import model_train, model_predict
from monitoring import monitor

app = Flask(__name__)


def params():
    return request.get_json(silent=True) or request.args


@app.route('/')
def index():
    return jsonify({'service': 'AAVAIL revenue forecast', 'endpoints': ['/train', '/predict', '/logs', '/monitor']})


@app.route('/train', methods=['GET', 'POST'])
def train():
    model_train(test=params().get('mode') == 'test')
    return jsonify({'trained': True})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    p = params()
    if not p.get('date'):
        return jsonify({'error': 'date (YYYY-MM-DD) is required'}), 400
    try:
        return jsonify(model_predict(p.get('country', 'all'), p['date'], test=p.get('mode') == 'test'))
    except (ValueError, FileNotFoundError) as e:
        return jsonify({'error': str(e)}), 400


@app.route('/logs')
def logs():
    return jsonify(sorted(os.listdir(LOG_DIR)) if os.path.isdir(LOG_DIR) else [])


@app.route('/logs/<filename>')
def log_file(filename):
    if not os.path.exists(os.path.join(LOG_DIR, filename)):
        return jsonify({'error': 'not found'}), 404
    return send_from_directory(LOG_DIR, filename)


@app.route('/monitor')
def drift():
    p = params()
    try:
        return jsonify(monitor(p.get('country', 'all'), test=p.get('mode') == 'test'))
    except (KeyError, FileNotFoundError) as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
