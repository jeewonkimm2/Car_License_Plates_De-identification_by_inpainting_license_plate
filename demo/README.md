## 💡 How to install & run API server

### 1. Activating virtual environment
``` bash
# Virtual environment creation
demo % python3 -m venv .venv

# Virtual environment activation
demo % source .venv/bin/activate

# Installation of necessary packages
(.venv) demo % pip install --upgrade pip
(.venv) demo % pip install Flask
(.venv) demo % pip install flask-cors

# Create empty folder to save images and videos
demo % mkdir database
demo % cd database && mkdir image video
```

### 2. Run server
``` bash
(.venv) demo % python app.py
```

<!-- TODO: How to use postman -->