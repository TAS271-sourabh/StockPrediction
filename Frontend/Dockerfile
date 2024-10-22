# Frontend/Dockerfile
# Use Node.js base image for building
FROM node:18-alpine as build

# Set the working directory for the build stage
WORKDIR /app

# Set the NODE_OPTIONS environment variable
ENV NODE_OPTIONS=--openssl-legacy-provider

# Copy package.json and package-lock.json from the interface directory
COPY interface/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code from the interface directory
COPY interface/ ./

# Build the React app
RUN npm run build

# Create a new stage to serve the built app
FROM node:18-alpine as production

# Set the working directory for the production image
WORKDIR /app

# Copy the built files from the build stage
COPY --from=build /app/build ./build

# Copy the server.js file from the interface directory
COPY interface/server.js .

# Install express and prom-client for the server
RUN npm install express prom-client

# Expose port 3001
EXPOSE 3001

# Command to run the server
CMD ["node", "server.js"]
