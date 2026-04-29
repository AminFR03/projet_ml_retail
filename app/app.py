from flask import Flask, request, jsonify, render_template
import sys
import os

# Add src to path so we can import predict module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import predict as pred_module

# Tell Flask where the templates folder lives (one level up from this file)
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_churn():
    try:
        data = request.get_json(force=True)
        model, scaler, feature_names = pred_module.load_resources()
        prediction, proba = pred_module.predict(data, model, scaler, feature_names)

        return jsonify({
            'prediction': int(prediction),
            'churn_risk': 'High' if prediction == 1 else 'Low',
            'probability': proba.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("=" * 55)
    print("  [Retail Churn Intelligence -- Flask Server]")
    print("=" * 55)

    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
    )
    if not os.path.exists(model_path):
        print("[!] Modele introuvable -- lancez preprocessing.py et train_model.py d'abord.")
    else:
        print("[+] Modele charge avec succes.")

    print("[*] Interface : http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)
