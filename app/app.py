import os

import gradio as gr
import torch
import torchvision.transforms as T
from torch.nn import Sequential
from torch.serialization import add_safe_globals

from scripts.utils.model import AlexNet

MODEL_VERSION = "{{MODEL_VERSION}}"
MODEL_DESCRIPTION = "{{MODEL_DESCRIPTION}}"

# dynamically get directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model_artifacts", "model.pth")

add_safe_globals([AlexNet, Sequential])

# load model
model = torch.load(
    MODEL_PATH,
    map_location=torch.device("cpu"),
    weights_only=False
)
model.eval()

# CIFAR-10 classes
classes = [
    "airplane", "automobile", "bird", "cat",
    "deer", "dog", "frog", "horse",
    "ship", "truck"
]

# Define transform same as test transforms
transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616])
])


def predict_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(image)
    preds = torch.softmax(logits, dim=1)
    confidences = {classes[i]: float(preds[0][i]) for i in range(len(classes))}
    return confidences


demo = gr.Interface(fn=predict_image,
                    inputs=gr.Image(type="pil"),
                    outputs=gr.Label(num_top_classes=10),
                    title=f"Image Classificator v{MODEL_VERSION}",
                    description=MODEL_DESCRIPTION,
                    )

demo.launch()
