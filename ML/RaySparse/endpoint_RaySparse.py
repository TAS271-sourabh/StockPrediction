from flask import Flask, request, jsonify, Response
from flask_cors import CORS  # Import CORS
from TrainModel import train_stock_model
import time
from Test_RaySparse import predict_price
from prometheus_client import Counter, Histogram, generate_latest  # Import Counter, Histogram, and generate_latest

app = Flask(__name__)
CORS(app)

# Define a Counter to track total request count
REQUEST_COUNT = Counter('request_count', 'Total request count', ['endpoint'])
# Define a Histogram to track training duration
TRAINING_DURATION = Histogram('training_duration_seconds', 'Duration of model training in seconds', ['stock_symbol'])
# Define a Histogram to track prediction duration
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Duration of prediction in seconds', ['stock_symbol'])

# Route to trigger model training
@app.route('/train', methods=['POST'])
def train_stock():
    stock_symbol = request.json.get('symbol')
    
    if not stock_symbol:
        return jsonify({"error": "No stock symbol provided"}), 400

    REQUEST_COUNT.labels(endpoint='/train').inc()  # Increment the request count for /train
    
    try:
        start_time = time.time()  # Start timing the training
        model_path = train_stock_model(stock_symbol)
        end_time = time.time()  # End timing the training

        training_time = end_time - start_time  # Calculate training time
        TRAINING_DURATION.labels(stock_symbol=stock_symbol).observe(training_time)  # Record the training duration

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

    REQUEST_COUNT.labels(endpoint='/predict').inc()  # Increment the request count for /predict
    
    try:
        start_time = time.time()  # Start timing the prediction
        # Get the prediction details for the stock and date
        prediction_details = predict_price(stock_symbol, date)
        end_time = time.time()  # End timing the prediction

        prediction_time = end_time - start_time  # Calculate prediction time
        PREDICTION_DURATION.labels(stock_symbol=stock_symbol).observe(prediction_time)  # Record the prediction duration
        
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
            "execution_time": prediction_details["execution_time"],
            "prediction_time": prediction_time  # Include prediction time in response
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/metrics') 
def metrics(): 
    return Response(generate_latest(), mimetype='text/plain')  # Return metrics in Prometheus format

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
