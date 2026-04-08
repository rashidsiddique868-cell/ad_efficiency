import sys
import os

# Add the parent directory to sys.path to import the main app
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
