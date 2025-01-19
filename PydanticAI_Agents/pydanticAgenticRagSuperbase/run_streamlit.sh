#!/bin/bash

# Kill any existing Streamlit processes
pkill -f streamlit

# Start Streamlit in the background
streamlit run streamlit_app.py --server.port 8501 &

# Wait a moment for Streamlit to start
sleep 2

# Print helpful information
echo "
========================================================
🚀 PDF Document Processing System is starting up!

📱 Access the UI at: http://localhost:8501

❌ To stop Streamlit, run: pkill -f streamlit

💡 If the UI doesn't open automatically, wait a few seconds 
   and try accessing the URL manually.
========================================================
" 