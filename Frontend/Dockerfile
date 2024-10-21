# Use Node.js base image
FROM node:18-alpine

# Set the working directory in the container
WORKDIR /interface

# Copy the package.json and package-lock.json (if present)
COPY interface/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY interface/ ./

# Build the app for production
RUN npm run build

# Install 'serve' to serve the build directory
RUN npm install -g serve

# Command to run the app
CMD ["serve", "-s", "build", "-l", "3000"]

# Expose port 3000
EXPOSE 3001
