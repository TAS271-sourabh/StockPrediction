const express = require('express');
const cors = require('cors');
const { collectDefaultMetrics, register, Gauge } = require('prom-client');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001; // Set to port 3001

app.use(cors()); // Enable CORS
app.use(express.json()); // Middleware to parse JSON bodies

// Collect default metrics (CPU, memory, etc.)
collectDefaultMetrics({ timeout: 5000 });

// Example custom metric
const frontendActionCount = new Gauge({
  name: 'frontend_action_count',
  help: 'Counts the number of actions taken on the frontend',
  labelNames: ['action_type'],
});

// Array to store frontend logs
let logs = [];

// Endpoint to receive logs from the frontend
app.post('/logs', (req, res) => {
  const { message } = req.body;
  if (message) {
    logs.push(message); // Store the log message
    console.log('Received log:', message);
    frontendActionCount.inc({ action_type: message }); // Increment metric
    res.status(200).json({ message: 'Log received successfully' });
  } else {
    res.status(400).json({ error: 'No log message provided' });
  }
});

// Endpoint to view the logs
app.get('/logs', (req, res) => {
  res.status(200).json(logs); // Return all logged messages
});

// Endpoint for Prometheus to scrape metrics
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

// Serve static files from the React app
app.use(express.static(path.join(__dirname, 'build')));

// Handle all other requests by sending the React app
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

// Start the server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
