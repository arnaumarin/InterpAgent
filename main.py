import subprocess
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "workspace",
        nargs="?",
        default="workspace",
        help="Workspace directory (default: ./workspace)"
    )
    args = parser.parse_args()

    abs_workspace = os.path.abspath(args.workspace)

    if not os.path.exists(abs_workspace):
        raise FileNotFoundError(f"❌ Workspace directory '{abs_workspace}' does not exist.")

    python = sys.executable
    subprocess.run([
        python, "-m",
        "streamlit", "run", "app.py",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ])

if __name__ == "__main__":
    main()
