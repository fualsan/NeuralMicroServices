import base64
import json
from urllib import request
import argparse

parser = argparse.ArgumentParser(description='Client application for image classifier service')

parser.add_argument('--img_path', help='path to image to be predicted', required=True)

args = parser.parse_args()


def read_and_encode_img(img_url):
    # encode image for sending over API
    with open(img_url, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image


def send_request(img_url):
    # Read image data from files

    encoded_img = read_and_encode_img(img_url)

    # Prepare the data to be sent
    data = {
        'image': encoded_img,
    }

    data = json.dumps(data).encode('utf-8')

    # send request to server
    # LOCAL
    URL = 'http://localhost:8000/image_classifier'
    req = request.Request(URL, data=data, headers={'Content-Type': 'application/json'})
    response = request.urlopen(req)

    # decode received image
    response_data = json.loads(response.read().decode('utf-8'))
    predicted_class = response_data['predicted_class']

    print(f'Predicted class: {predicted_class}')


if __name__ == '__main__':
    send_request(args.img_path)
