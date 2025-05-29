# Stage 1: Build the Next.js application
FROM node:18-slim AS builder
WORKDIR /app
COPY webapp/package.json webapp/package-lock.json ./webapp/
RUN cd webapp && npm install
COPY webapp ./webapp
RUN cd webapp && npm run build

# Stage 2: Setup Python environment and final application
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies that might be needed by Python packages
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python application source code
COPY src ./src
COPY app.py .
# Copy any other necessary files like .env if they were used (currently not)

# Copy the built Next.js app from the builder stage
COPY --from=builder /app/webapp/.next ./webapp/.next
COPY --from=builder /app/webapp/public ./webapp/public
COPY --from=builder /app/webapp/package.json ./webapp/package.json
COPY --from=builder /app/webapp/next.config.js ./webapp/next.config.js
# If next.config.mjs is used, copy that instead of .js
# COPY --from=builder /app/webapp/next.config.mjs ./webapp/next.config.mjs

# Expose the port Next.js runs on
EXPOSE 3000

# Set the command to run the Next.js application
# Note: The Next.js app needs to be able to find and execute python3 app.py
# Ensure python3 is in the PATH and app.py is executable if needed.
# The API route in Next.js already uses 'python3'.
ENV PYTHONUNBUFFERED=1
CMD ["npm", "start", "--prefix", "webapp"]
