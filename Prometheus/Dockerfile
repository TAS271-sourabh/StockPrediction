# Use the official Prometheus image from Docker Hub
FROM prom/prometheus:latest

# Copy the prometheus.yml configuration file into the container
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Expose the default Prometheus port
EXPOSE 9090

# Set the command to run Prometheus with the specified configuration file
CMD [ "--config.file=/etc/prometheus/prometheus.yml" ]
