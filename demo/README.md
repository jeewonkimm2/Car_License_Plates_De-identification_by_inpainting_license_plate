## 💡 How to install & run API server

### 1. Activating virtual environment
``` bash
# Virtual environment creation
demo % python3 -m venv .venv

# Virtual environment activation
demo % source .venv/bin/activate

# Installation of necessary packages
(.venv) demo % pip install --upgrade pip
# Necessary packages for API runs
(.venv) demo % pip install Flask
(.venv) demo % pip install flask-cors
# Necessary packages for liscense plate detection model
(.venv) demo % pip install roboflow
(.venv) demo % pip install opencv-python
# Necessary packages for image generation model
(.venv) demo % pip install -r requirements.txt

# Create empty folder to save images
demo % mkdir database
demo % cd database && mkdir image
```

### 2. Run server
``` bash
(.venv) demo % python app.py
```