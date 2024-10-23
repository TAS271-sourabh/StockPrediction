import React, { useEffect, useState } from 'react';
import './Dashboard.css';

const Dashboard = () => {
  // Predefined list of 10 Nifty 50 stocks
  const stocksList = [
    'RELIANCE',
    'HDFCBANK',
    'INFY',
    'TCS',
    'HINDUNILVR',
    'ITC',
    'KOTAKBANK',
    'LT',
    'SBIN',
    'HDFC'
  ];

  const [stocks, setStocks] = useState(stocksList); // Set initial stocks to the predefined list
  const [selectedStock, setSelectedStock] = useState('');
  const [predictionDate, setPredictionDate] = useState('');
  const [predictionResults, setPredictionResults] = useState(null);
  const [trainingMessage, setTrainingMessage] = useState('');
  const [isTrained, setIsTrained] = useState(false); // Track if stock is trained
  const [isLoading, setIsLoading] = useState(false); // Track loading state for training

  // Function to log frontend actions
  const logFrontendAction = async (actionType) => {
    try {
      await fetch('http://localhost:3000/logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: `Action performed: ${actionType}` }),
      });
    } catch (error) {
      console.error('Error logging action:', error);
    }
  };

  const handleTrain = async () => {
    if (!selectedStock) return;

    setIsLoading(true); // Set loading to true while training
    const response = await fetch('http://localhost:5000/train', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ symbol: selectedStock }),
    });

    const data = await response.json();
    setIsLoading(false); // Set loading to false after training completes

    if (response.ok) {
      setTrainingMessage(`Stock ${selectedStock} has been trained successfully!`);
      setIsTrained(true); // Set stock as trained
      logFrontendAction('train_success'); // Log successful training
    } else {
      setTrainingMessage(data.error);
      logFrontendAction('train_failure'); // Log failed training
    }
  };

  const handlePredict = async () => {
    if (!selectedStock || !predictionDate) return;

    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ stock_symbol: selectedStock, date: predictionDate }), // Ensure correct payload
    });

    const data = await response.json();
    if (response.ok) {
      setPredictionResults(data);
      logFrontendAction('predict_success'); // Log successful prediction
    } else {
      setPredictionResults({ error: data.error });
      logFrontendAction('predict_failure'); // Log failed prediction
    }
  };

  return (
    <div className="dashboard-container">
      <h1>Stock Prediction Dashboard</h1>
      <div className="input-group">
        <label htmlFor="stocks">Select a Stock:</label>
        <select id="stocks" value={selectedStock} onChange={(e) => setSelectedStock(e.target.value)}>
          <option value="">--Select a Stock--</option>
          {stocks.map((stock) => (
            <option key={stock} value={stock}>{stock}</option>
          ))}
        </select>
        <button 
          onClick={handleTrain} 
          className="train-button" 
          disabled={isLoading} // Disable during loading
        >
          {isLoading ? 'Training...' : `Train Stock ${selectedStock}`}
        </button>
        {trainingMessage && <p className="training-message">{trainingMessage}</p>}
      </div>
      <div className="input-group">
        <label htmlFor="date">Select Date for Prediction:</label>
        <input type="date" id="date" onChange={(e) => setPredictionDate(e.target.value)} />
        <button 
          onClick={handlePredict} 
          className="predict-button" 
          disabled={!isTrained} // Disable until training is done
          style={{ backgroundColor: !isTrained ? '#ccc' : '#28a745' }} // Change color based on state
        >
          Predict Price
        </button>
      </div>
      {predictionResults && (
        <div className="results">
          {predictionResults.error ? (
            <p className="error">{predictionResults.error}</p>
          ) : (
            <div className="prediction-details">
              <h2>Prediction Results for {predictionResults.stockSymbol} on {predictionResults.date}</h2>
              <p>Actual Open: {predictionResults.actual_open}</p>
              <p>Actual Close: {predictionResults.actual_close}</p>
              <p>Predicted Open: {predictionResults.predicted_open}</p>
              <p>Predicted Close: {predictionResults.predicted_close}</p>
              <p>Open Accuracy: {predictionResults.open_accuracy}</p>
              <p>Close Accuracy: {predictionResults.close_accuracy}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Dashboard;
