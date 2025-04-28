import subprocess
import sys
import os
import time

def print_management_commands():
    print("\n" + "="*50)
    print("Streamlit Management Commands:")
    print("="*50)
    print("# View all Streamlit processes")
    print("ps -aef | grep streamlit")
    print("\n# Kill all Streamlit processes")
    print("pkill -f streamlit")
    print("="*50 + "\n")

def run_streamlit():
    try:
        # Start Streamlit in a subprocess
        process = subprocess.Popen(
            ["streamlit", "run", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait a bit to ensure Streamlit starts
        time.sleep(2)
        
        # Check if the process started successfully
        if process.poll() is None:
            print("\n✅ Streamlit app is running!")
            print("🌐 Open your browser to the URL shown above")
            print_management_commands()
        else:
            stdout, stderr = process.communicate()
            print("❌ Failed to start Streamlit:")
            print(stderr)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down Streamlit...")
        process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🚀 Starting Cisco RAG System...")
    run_streamlit() 