import torchvision
import torch.nn as nn
import torch.nn.functional as F
import glob
from PIL import Image
from torchvision import transforms
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import fit
import os
import cv2 as cv
import numpy as np

# Image transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # Channel first, normalization, convert to a tensor
])

class dataset(data.Dataset):
    def __init__(self, imgs_path):
        self.imgs_path = imgs_path

    def __getitem__(self, item):  # Implement the slicing method
        img = self.imgs_path[item]

        pil_img = Image.open(img)
        img_tensor = transform(pil_img)

        return img_tensor

    def __len__(self):
        return len(self.imgs_path)

# Create the model
class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class OurModel(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        e1 = self.layer1(input) # 64,128,128
        e2 = self.layer2(e1) # 64,64,64
        e3 = self.layer3(e2) # 128,32,32
        e4 = self.layer4(e3) # 256,16,16
        f = self.layer5(e4) # 512,8,8
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        return out

model = OurModel(2)
PATH = 'OurModel.pth'
model.load_state_dict(torch.load(PATH))  # Restore model weights

# Prediction application
path = './data/train/img/39_orig.jpg'
pil_img = Image.open(path)
img_tensor = transform(pil_img)  # Convert type
img_tensor_batch = torch.unsqueeze(img_tensor, dim=0)  # Add a dimension of size 1
pred = model(img_tensor_batch)
pred = torch.argmax(pred[0].permute(1, 2, 0), axis=-1).detach().numpy()
plt.subplot(1, 2, 1)
plt.imshow(img_tensor.permute(1, 2, 0).numpy())
plt.subplot(1, 2, 2)
plt.imshow(pred)
plt.show()

pred = np.where(pred == 1, 255, pred)
cv.imwrite("./data/output.png", pred)
pred = cv.imread("./data/output.png", 1)
height, width, channels = pred.shape

# Iterate through each pixel of each channel
for y in range(height):
    for x in range(width):
        # Get BGR values
        b, g, r = pred[y, x]
        b = 0
        g = 0
        pred[y, x] = (b, g, r)

# Overlay prediction and original image
src_image = cv.imread(path, 1)
src_image = cv.resize(src_image, (512, 512), interpolation=cv.INTER_AREA)

# Use cv2.addWeighted for overlay
result = cv.addWeighted(src_image, 0.4, pred, 0.6, 0)

# Display the result
cv.imshow('Result', result)
cv.waitKey(0)  # Wait for a key press, 0 means indefinitely
cv.destroyAllWindows()  