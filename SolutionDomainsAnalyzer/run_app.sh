#!/bin/bash

# Start Streamlit server in the background
streamlit run app.py --server.maxUploadSize 1024 &

# Wait a moment for the server to start
sleep 2

# Open the default browser (works on macOS)
open http://localhost:8501

# Display helpful messages
echo "----------------------------------------"
echo "🚀 Streamlit server started!"
echo "📱 Access the app at: http://localhost:8501"
echo "----------------------------------------"
echo "⚠️  To stop the server, use: pkill -f streamlit"
echo "----------------------------------------" 