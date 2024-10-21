# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory inside the container to /ML/RaySparse
WORKDIR /ML/RaySparse

# Copy everything from the ML directory into /ML in the container
COPY . /ML

# Install the required Python packages from the RaySparse directory
RUN pip install --no-cache-dir -r /ML/requirements.txt

# Expose the port on which your application will run
EXPOSE 5000

# Run the endpoint script as a server
CMD ["python3", "endpoint_RaySparse.py"]
