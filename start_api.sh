#!/bin/bash

# Start Mainframe Agent API Server

echo "Starting Mainframe Agent API Server..."
echo "API will be available at: http://localhost:8001"
echo "Swagger UI: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the API server
python api_server.py
