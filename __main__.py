
import os, sys, subprocess
def main():
    # Run the Streamlit hub app
    here = os.path.dirname(__file__)
    app = os.path.join(here, "main_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app, "--server.port=8501", "--server.headless=true"])
