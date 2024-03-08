import torch
from torchvision.models import resnet18, ResNet18_Weights

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64
import uvicorn
import json


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


with open('imagenet_class_index.json') as labels_file:
    labels = json.load(labels_file)


weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights, progress=False).eval()
#model = torch.jit.script(model).to(device)
model = model.to(device)
transforms = weights.transforms(antialias=True)


@torch.no_grad()
def predict(img):
    img = transforms(img).to(device)
    pred = model(img.unsqueeze(0))
    # (batch_size, num_classes)
    # num_classes 1000 is for Imagenet
    pred = pred.argmax(dim=1).item()
    # ex: ["n01440764", "tench"]
    pred_str = labels[str(pred)][1]
    return pred_str


# uvicorn runs this
app = FastAPI()


class ImageClassifierRequest(BaseModel):
	image: str # base64 encoded image as string
     

@app.post('/image_classifier')
async def process_image(request: ImageClassifierRequest):

    # decode base64 encoded low res image
    decoded_image = base64.b64decode(request.image, validate=True)
    decoded_image = Image.open(BytesIO(decoded_image))

    pred_str = predict(decoded_image)

    return {'predicted_class': pred_str}


if __name__ == '__main__':
	# optionally run from terminal: uvicorn image_classifier_server:app --host 0.0.0.0 --port 8000 --reload
	# accept every connection (not only local connections)
    uvicorn.run(app, host='0.0.0.0', port=8000)