import subprocess
import sys

# Example of detaching the process
if len(sys.argv) == 1:
    # Start a new instance of this script with a command line argument to prevent recursion
    subprocess.Popen([sys.executable, __file__, 'run_in_background'])
    sys.exit()
else:
    # Your existing code
    from threading import Thread
    import os

def run_streamlit():
  os.system('streamlit run app_adopt_barrier.py --server.port 8501')
  
thread = Thread(target=run_streamlit)
thread.start()  