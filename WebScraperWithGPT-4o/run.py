import subprocess
import sys
import time
import webbrowser
import socket
from pathlib import Path

def find_available_port(start=8501, end=8599):
    """Find an available port in the given range."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except socket.error:
                continue
    return None

def run_streamlit_server():
    """Run the Streamlit server in a subprocess."""
    # Get the directory of the current script
    current_dir = Path(__file__).parent.absolute()
    app_path = current_dir / 'app.py'

    # Find an available port
    port = find_available_port()
    if not port:
        print("Error: No available ports found")
        sys.exit(1)
    
    # Start Streamlit in a subprocess
    cmd = f"streamlit run {app_path} --server.port {port} --server.headless true"
    subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    # Open browser
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    
    print(f"\nStreamlit server started at {url}")
    print("\nProcess Management Commands:")
    print("----------------------------")
    print("To view Streamlit processes: ps -aef | grep streamlit")
    print("To kill all Streamlit processes: pkill -f streamlit")
    print("Your terminal is now available for use.")

if __name__ == "__main__":
    run_streamlit_server() 