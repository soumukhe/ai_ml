#!/bin/bash

# Run Streamlit in the background
python -m streamlit run pdf_runStreamlit.py > /dev/null 2>&1 &

# Wait a moment for Streamlit to start
sleep 2

# Print instructions
echo -e "\n\033[1mStreamlit server is starting in the background!\033[0m"
echo -e "\033[32mTo access the UI, open your browser and go to: \033[1mhttp://localhost:8501\033[0m"
echo -e "\033[33mTo kill the Streamlit server, run: \033[1mpkill -f streamlit\033[0m\n" 