from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from TrainModel import train_stock_model
import time
from Test_RaySparse import predict_price

app = Flask(__name__)
CORS(app) 

# Route to trigger model training
@app.route('/train', methods=['POST'])
def train_stock():
    # Get the stock symbol from the request
    stock_symbol = request.json.get('symbol')
    
    if not stock_symbol:
        return jsonify({"error": "No stock symbol provided"}), 400

    # Call the train.py script with the stock symbol
    try:
        start_time = time.time()  # Start timing the training
        model_path = train_stock_model(stock_symbol)
        end_time = time.time()  # End timing the training

        training_time = end_time - start_time  # Calculate training time

        return jsonify({
            "message": f"Model trained for {stock_symbol}",
            "model_path": model_path,
            "training_time": training_time  # Include training time in response
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_stock_details():
    date = request.json.get('date')
    stock_symbol = request.json.get('stock_symbol')  # Change to match the correct key in your JSON

    if not stock_symbol:
        return jsonify({"error": "No stock symbol provided"}), 400

    if not date:
        return jsonify({"error": "No date provided"}), 400

    try:
        # Get the prediction details for the stock and date
        prediction_details = predict_price(stock_symbol, date)
        
        # Return the prediction details
        return jsonify({
            "stockSymbol": stock_symbol,
            "date": date,
            "actual_open": prediction_details["actual_open"],
            "actual_close": prediction_details["actual_close"],
            "predicted_open": prediction_details["predicted_open"],
            "predicted_close": prediction_details["predicted_close"],
            "open_accuracy": prediction_details["open_accuracy"],
            "close_accuracy": prediction_details["close_accuracy"],
            "execution_time": prediction_details["execution_time"]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
