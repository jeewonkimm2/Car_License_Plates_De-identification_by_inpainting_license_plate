from flask import Flask, request, send_file
from flask_cors import CORS
from io import BytesIO
import os
from roboflow import Roboflow
import cv2
import numpy as np
import subprocess

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Allowed extensions for image format
ALLOWED_EXTENSIONS_IMAGE = {'png'}

DB_PATH = 'database'
IMAGE_PATH = os.path.join(DB_PATH, 'image')

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
            x1, x2 = int(x - size), int(x + size)
            y1, y2 = int(y - size), int(y + size)
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
            
            # Generation
            border = 256
            center = border/2
            x1 = int(center - width/2) if center - width/2 >= 0 else 0
            y1 = int(center - height/2) if center - height/2 >= 0 else 0
            x2 = int(center + width/2) if center + width/2 <= border else border
            y2 = int(center + height/2) if center + height/2 <= border else border
            generated_path = os.path.join(current_image_root, input_file.filename.split(".")[0]+"_generated."+input_file.filename.split(".")[-1])
            command = [
                'python', 'generate.py',
                '--image', cropped_path,
                '--output', generated_path,
                '--x1', f'{x1}', '--x2', f'{x2}', '--y1', f'{y1}', '--y2', f'{y2}'
            ]
            try:
                subprocess.run(command, check=True)
                app.logger.info(f'Image generated: {generated_path}')
            except subprocess.CalledProcessError as e:
                print(f"Error on image generation: {e}")
                
            # Output
            original_image = cv2.imread(current_image_path)
            generated_image = cv2.imread(generated_path)
            
            generated_height, generated_width, _ = generated_image.shape
            start_x = x - generated_width // 2
            start_y = y - generated_height // 2
            generated_part = generated_image[:int(min(generated_height, original_image.shape[0] - start_y)),
                                :int(min(generated_width, original_image.shape[1] - start_x))]
            original_image[int(start_y):int(start_y + generated_part.shape[0]), int(start_x):int(start_x + generated_part.shape[1])] = generated_part
            final_path = os.path.join(current_image_root, input_file.filename.split(".")[0]+"_output."+input_file.filename.split(".")[-1])
            cv2.imwrite(final_path, original_image) # Saving final image
            
            return input_file.filename
        else:
            return 'Try with another filename'
    
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
        image_file = os.path.join(image_file_root, filename.split(".")[0]+"_output."+filename.split(".")[-1])
        image_type = f'image/{filename.split(".")[-1]}'

        return send_file(image_file, mimetype=image_type)
    
    else:
        return 'Unsupported input format', 400

if __name__ == '__main__':
    app.run(port='5001', debug=True)