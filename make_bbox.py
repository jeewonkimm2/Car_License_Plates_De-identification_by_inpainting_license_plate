from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import os

# 각 데이터셋 annotation 표기 방법이 달라 따로 처리

# 데이터셋 1) 캐글(https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) 데이터에 대한 화이트 bbox 처리
def draw_bbox_on_image(image_path, xml_path, output_path):
    # Load image
    image = Image.open(image_path)

    # Load XML data
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get bounding box coordinates
    xmin = int(root.find(".//bndbox/xmin").text)
    ymin = int(root.find(".//bndbox/ymin").text)
    xmax = int(root.find(".//bndbox/xmax").text)
    ymax = int(root.find(".//bndbox/ymax").text)

    # Draw bounding box on the image
    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="white", width=3)

    # Fill the bounding box with white color
    draw.rectangle([xmin, ymin, xmax, ymax], fill="white")

    # Save the image with the drawn bounding box
    image.save(output_path)

# 데이터셋 2) 로보플로우(https://public.roboflow.com/object-detection/license-plates-us-eu) 데이터에 대한 화이트 bbox 처리
def draw_white_rectangle_on_images(image_folder, coordinates_folder, output_folder):
    # 이미지 폴더와 좌표 폴더에서 파일 리스트 가져오기
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    coordinates_files = [f for f in os.listdir(coordinates_folder) if f.endswith('.txt')]

    # 좌표 파일과 이미지 파일을 매칭하여 처리
    for coordinates_file in coordinates_files:
        # 좌표 파일 이름과 일치하는 이미지 파일 찾기
        image_file = next((f for f in image_files if f.startswith(coordinates_file.split('.')[0])), None)

        if image_file:
            # 이미지 열기
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path)

            # 좌표 파일 열기
            coordinates_path = os.path.join(coordinates_folder, coordinates_file)
            with open(coordinates_path, 'r') as coord_file:
                # 좌표 읽기
                line = coord_file.readline()
                parts = line.split()
                class_label = int(parts[0])
                x, y, w, h = [float(coord) for coord in parts[1:5]]

                # 좌표를 이미지에 그리기
                draw = ImageDraw.Draw(image)
                # image_size = image.size
                # x, y, w, h = [int(coord * dim) for coord, dim in zip([x, y, w, h], image_size)]
                draw.rectangle([x, y, w, h], fill="white")

                # 결과 이미지 저장
                output_path = os.path.join(output_folder, f"{image_file.split('.')[0]}_annotated.png")
                image.save(output_path)







if __name__ == "__main__":
    
    # 데이터셋1 처리
    # 이미지 폴더, 좌표 폴더, 결과 저장 폴더 지정
    input_folder = "./archive/images/"
    xml_folder = "./archive/annotations/"
    output_folder = "./archive/masked_images"

    os.makedirs(output_folder, exist_ok=True)

    for xml_filename in os.listdir(xml_folder):
        if xml_filename.endswith('.xml'):
            # XML 파일 경로
            xml_path = os.path.join(xml_folder, xml_filename)

            # 이미지 파일 경로
            image_filename = os.path.splitext(xml_filename)[0] + ".png"
            image_path = os.path.join(input_folder, image_filename)

            # 결과 이미지 파일 경로
            output_filename = os.path.splitext(xml_filename)[0] + "_with_bbox.png"
            output_path = os.path.join(output_folder, output_filename)

            # 이미지에 bbox 그리고 저장
            draw_bbox_on_image(image_path, xml_path, output_path)
    
    
    # 데이터셋2 처리
    # 이미지 폴더, 좌표 폴더, 결과 저장 폴더 지정
    image_folder_path = './archive2/images' # 이미지 폴더 위치
    coordinates_folder_path = './archive2/annotations' # 좌표 폴더 위치
    output_folder_path = './archive2/masked_images' # 이미지에 흰색 사각형 추가 후 저장될 폴더 위치

    # 결과 저장 폴더가 없으면 생성
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 함수 호출
    draw_white_rectangle_on_images(image_folder_path, coordinates_folder_path, output_folder_path)

