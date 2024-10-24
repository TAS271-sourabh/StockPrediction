# Stage 1: Build the React app
FROM node:18 AS build

# Set the working directory
WORKDIR /app

# Set legacy OpenSSL support to avoid the ERR_OSSL_EVP_UNSUPPORTED issue
ENV NODE_OPTIONS=--openssl-legacy-provider

# Copy the package.json and package-lock.json files from the interface directory
COPY interface/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code from the interface directory
COPY interface/ ./

# Build the React app
RUN npm run build

# Stage 2: Serve the app and run the server
FROM node:18

# Set the working directory
WORKDIR /app

# Copy the build output from the first stage
COPY --from=build /app/build ./build

# Copy the server.js file (it's in the same directory as the Dockerfile)
COPY server.js ./

# Copy package.json and package-lock.json for the server from the interface directory
COPY interface/package*.json ./

# Install server dependencies
RUN npm install

# Set legacy OpenSSL support for runtime as well
ENV NODE_OPTIONS=--openssl-legacy-provider

# Expose the port for the server
EXPOSE 3001

# Command to run the server
CMD ["node", "server.js"]
