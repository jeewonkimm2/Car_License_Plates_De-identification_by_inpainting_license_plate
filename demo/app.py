from flask import Flask, request, send_file
from flask_cors import CORS
from io import BytesIO
import os

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# TODO: Update allowed extensions for each image/video format
ALLOWED_EXTENSIONS_IMAGE = {'png'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4'}

DB_PATH = 'database'
IMAGE_PATH = os.path.join(DB_PATH, 'image')
VIDEO_PATH = os.path.join(DB_PATH, 'video')

def get_file_type(filename):
    if filename is not None:
        format = filename.split('.')[-1]
        if format in ALLOWED_EXTENSIONS_IMAGE:
            return 'Image'
        elif format in ALLOWED_EXTENSIONS_VIDEO:
            return 'Video'
        else:
            return 'Unsupported'
    else:
        return 'Empty'

# API test request.
# If you'd like to test the server is running,
# please find http://127.0.0.1:5000/hello
# If server is not running, please run this file. (`python app.py`)
@app.route('/hello', methods=['GET'])
def hello():
    return "hello"

@app.route('/upload', methods=['POST'])
def upload():
    # Validity check
    if 'input_file' not in request.files:
        app.logger.error('Error detected on request parameter')
        return 'No input_file part', 400
    
    input_file = request.files['input_file']
    # If the user does not select a file, the browser may submit an empty file without a filename
    if input_file.filename == '':
        app.logger.error('Input file is empty')
        return 'No selected file', 400
    
    app.logger.info(f'{input_file.filename} has been uploaded')

    if get_file_type(input_file.filename) == "Image":

        current_image_root = os.path.join(IMAGE_PATH, input_file.filename.split(".")[0])

        if not os.path.exists(current_image_root):
            os.makedirs(current_image_root)

            current_image_path = os.path.join(current_image_root, input_file.filename)
            input_file.save(current_image_path)
            
            # TODO : Handle uploaded image
            app.logger.info(f'Image file processing ...')

            return input_file.filename
        else:
            return 'Try with another filename'

    elif get_file_type(input_file.filename) == "Video":
        app.logger.info(f'Video file processing ...')

        # TODO : Save uploaded video
        # TODO : Handle uploaded video
        
    else:
        return 'Unsupported input format', 400
    
    return "Succeed"

@app.route('/output', methods=['GET'])
def output():
    output_data = request.get_json()
    print(output_data)
    filename = output_data["filename"] if "filename" in output_data else None

    if get_file_type(filename) == "Image":
        app.logger.info(f'Returning image file ...')

        image_file_root = os.path.join(IMAGE_PATH, filename.split(".")[0])
        image_file = os.path.join(image_file_root, filename)
        image_type = f'image/{filename.split(".")[-1]}'

        return send_file(image_file, mimetype=image_type)

    elif get_file_type(filename) == "Video":
        app.logger.info(f'Returning video file ...')

        return "Proceeded video"
        
    else:
        return 'Unsupported input format', 400

if __name__ == '__main__':
    app.run(port='5001', debug=True)