from flask import Flask, request, send_file
from flask_cors import CORS
from io import BytesIO
import os
from roboflow import Roboflow
import cv2
import numpy as np

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# TODO: Update allowed extensions for each image/video format
ALLOWED_EXTENSIONS_IMAGE = {'png'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4'}

DB_PATH = 'database'
IMAGE_PATH = os.path.join(DB_PATH, 'image')
VIDEO_PATH = os.path.join(DB_PATH, 'video')

# lisence-plate detection model(pretrained)
ROBOFLOW_API_KEY = "" # Insert api key here
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("vehicle-registration-plates-trudk")
model = project.version(2).model

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
            
            # Liscence plate detection
            lisence_plate_info = model.predict(current_image_path, confidence=40, overlap=30).json()
            app.logger.info(f'Car plate detected: \n {lisence_plate_info}')
            
            predicted_path = os.path.join(current_image_root, input_file.filename.split(".")[0]+"_predicted."+input_file.filename.split(".")[-1])
            model.predict(current_image_path, confidence=40, overlap=30).save(predicted_path)
            
            x = lisence_plate_info['predictions'][0]['x']
            y = lisence_plate_info['predictions'][0]['y']
            width = lisence_plate_info['predictions'][0]['width']
            height = lisence_plate_info['predictions'][0]['height']
            
            # Masking
            image = cv2.imread(current_image_path)
            img_height, img_width, img_channels = image.shape
            app.logger.info(f'Given image: height-{img_height}, width-{img_width}, channel{img_channels}')
            
            x1 = int(x - width/2) if x - width/2 >= 0 else 0
            y1 = int(y - height/2) if y - height/2 >= 0 else 0
            x2 = int(x + width/2) if x + width/2 <= img_width else img_width
            y2 = int(y + height/2) if y + height/2 <= img_height else img_height
            
            mask = np.zeros_like(image)
            color=(128, 128, 130)
            image[y1:y2, x1:x2] = color
            
            masked_path = os.path.join(current_image_root, input_file.filename.split(".")[0]+"_masked."+input_file.filename.split(".")[-1])
            cv2.imwrite(masked_path, image) # Saving masked image
            
            size = 128 # 256 / 2
            x1, x2 = x - size, x + size
            y1, y2 = y - size, y + size
            if x1 < 0:
                image = cv2.copyMakeBorder(image, 0, 0, abs(x1), 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                x2 += abs(x1)
                x1 = 0
            if y1 < 0:
                image = cv2.copyMakeBorder(image, abs(y1), 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                y2 += abs(y1)
                y1 = 0
            if x2 > image.shape[1]:
                image = cv2.copyMakeBorder(image, 0, 0, 0, x2 - image.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            if y2 > image.shape[0]:
                image = cv2.copyMakeBorder(image, 0, y2 - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            cropped_image = image[y1:y2, x1:x2]
            cropped_path = os.path.join(current_image_root, input_file.filename.split(".")[0]+"_cropped."+input_file.filename.split(".")[-1])
            cv2.imwrite(cropped_path, cropped_image) # Saving cropped image

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