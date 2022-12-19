import torch
from torch import nn
import gradio as gr

# Load the model and then the post-training state_dict
model = torch.load('mnist_model.pth',map_location=torch.device('cpu'))
model.load_state_dict(torch.load('mnist_model_weights.pth',map_location=torch.device('cpu')))
model.eval()

# Prediction function
def predict(img):
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
    with torch.no_grad():
        pred = model(x)[0]
    return int(pred.argmax())

# Define and launch gradio interfact with sketchopad input and classification label output
title = "Guess that digit"
description = "Draw your favorite base-10 digit (0-9) and click submit - I'll try to guess what you drew! I do a bit better if you're not too messy and your digit is fairly centered."
gr.Interface(fn=predict, 
             inputs="sketchpad",
             outputs="label",
             title = title,
             description = description,
              ).launch()