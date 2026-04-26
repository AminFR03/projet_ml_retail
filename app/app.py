from flask import Flask, request, jsonify, render_template_string
import sys
import os

# Add src to path so we can import predict module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import predict as pred_module

app = Flask(__name__)

# Beautiful interactive HTML template for UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retail Churn - Intelligence Artificielle</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg-color: #0f172a;
            --panel-bg: rgba(30, 41, 59, 0.7);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --danger: #ef4444;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            background: var(--bg-color);
            background-image: radial-gradient(circle at top right, #334155 0%, transparent 40%),
                              radial-gradient(circle at bottom left, #1e1b4b 0%, transparent 40%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .container {
            width: 100%;
            max-width: 500px;
            padding: 40px;
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            transition: transform 0.3s ease;
        }
        
        .container:hover {
            transform: translateY(-5px);
        }

        h1 {
            font-weight: 800;
            font-size: 28px;
            margin-top: 0;
            margin-bottom: 10px;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }

        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 30px;
            font-size: 14px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-size: 13px;
            font-weight: 600;
            color: #cbd5e1;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input {
            width: 100%;
            padding: 14px 16px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: white;
            font-size: 16px;
            box-sizing: border-box;
            transition: all 0.2s ease;
            font-family: 'Inter', sans-serif;
        }

        input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        button {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 10px;
            box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
        }

        button:hover {
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 15px 20px -3px rgba(99, 102, 241, 0.4);
        }
        
        button:active {
            transform: translateY(1px);
        }

        #result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            font-size: 18px;
            opacity: 0;
            transform: scale(0.95);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: none;
        }
        
        .result-visible {
            opacity: 1 !important;
            transform: scale(1) !important;
            display: block !important;
        }

        .risk-high { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
        .risk-low { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
        
        .probability-bar {
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            margin-top: 12px;
            overflow: hidden;
            position: relative;
        }
        
        .probability-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 1s ease-out;
            width: 0%;
        }

        .flex-row {
            display: flex;
            gap: 15px;
        }
        .flex-row .form-group {
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Predictif Churn Client</h1>
        <p class="subtitle">Analysez le comportement et prévenez le départ</p>
        
        <form id="predictionForm">
            <div class="flex-row">
                <div class="form-group">
                    <label>Récence (Jours)</label>
                    <input type="number" id="recency" value="15" required min="0">
                </div>
                <div class="form-group">
                    <label>Fréquence d'achat</label>
                    <input type="number" id="frequency" value="3" required min="1">
                </div>
            </div>
            
            <div class="flex-row">
                <div class="form-group">
                    <label>Montant Total (£)</label>
                    <input type="number" id="monetaryTotal" value="150" step="0.1" required>
                </div>
                <div class="form-group">
                    <label>Ancienneté (Jours)</label>
                    <input type="number" id="tenure" value="365" required min="1">
                </div>
            </div>

            <button type="submit" id="submitBtn">Analyser le risque de départ</button>
        </form>

        <div id="result">
            <div id="resultText"></div>
            <div class="probability-bar">
                <div id="probFill" class="probability-fill"></div>
            </div>
            <div style="font-size: 12px; margin-top: 8px; font-weight: 400; opacity: 0.8;" id="probDetails"></div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('submitBtn');
            btn.innerHTML = 'Analyse en cours...';
            btn.style.opacity = '0.7';
            
            // Get base values
            const recency = parseFloat(document.getElementById('recency').value);
            const frequency = parseFloat(document.getElementById('frequency').value);
            const monetaryTotal = parseFloat(document.getElementById('monetaryTotal').value);
            const customerTenure = parseFloat(document.getElementById('tenure').value);
            
            // Calculate derived features like we did in python
            const monetaryPerDay = monetaryTotal / (recency + 1);
            const avgBasketValue = frequency > 0 ? monetaryTotal / frequency : 0;
            const tenureRatio = customerTenure > 0 ? recency / customerTenure : 0;

            const payload = {
                Recency: recency,
                Frequency: frequency,
                MonetaryTotal: monetaryTotal,
                MonetaryPerDay: monetaryPerDay,
                AvgBasketValue: avgBasketValue,
                CustomerTenure: customerTenure,
                TenureRatio: tenureRatio
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                const resultDiv = document.getElementById('result');
                const resultText = document.getElementById('resultText');
                const probFill = document.getElementById('probFill');
                const probDetails = document.getElementById('probDetails');
                
                resultDiv.className = data.prediction === 1 ? 'risk-high result-visible' : 'risk-low result-visible';
                
                if(data.prediction === 1) {
                    resultText.innerHTML = '⚠️ Risque de départ ÉLEVÉ';
                    probFill.style.background = '#ef4444';
                    probFill.style.width = (data.probability[1] * 100) + '%';
                    probDetails.innerHTML = `Probabilité de départ: ${(data.probability[1] * 100).toFixed(1)}%`;
                } else {
                    resultText.innerHTML = '✅ Client FIDELE';
                    probFill.style.background = '#10b981';
                    probFill.style.width = (data.probability[0] * 100) + '%';
                    probDetails.innerHTML = `Probabilité de rester: ${(data.probability[0] * 100).toFixed(1)}%`;
                }
                
            } catch (err) {
                alert("Erreur de connexion à l'API");
            } finally {
                btn.innerHTML = 'Analyser le risque de départ';
                btn.style.opacity = '1';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

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
    print("Starting Flask application...")
    # Check if models exist
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl'))
    if not os.path.exists(model_path):
        print("Warning: Models not found. Please run src/preprocessing.py and src/train_model.py first.")
    app.run(debug=True, host='0.0.0.0', port=5000)
