import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
import glob
from PIL import Image
import fit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load training set images and annotations
image_folder = './data/train/img'
mask_folder = './data/train/mask'
images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
anno = sorted(glob.glob(os.path.join(mask_folder, '*.png')))

# Load test set images and annotations
test_image_folder = './data/test/img'
test_anno_folder = './data/test/mask'
test_images = sorted(glob.glob(os.path.join(test_image_folder, '*.jpg')))
test_anno = sorted(glob.glob(os.path.join(test_anno_folder, '*.png')))

# Define the transformation
transform = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),  
])

# Define a custom dataset class
class dataset(data.Dataset):
    def __init__(self, imgs_path, annos_path):
        self.imgs_path = imgs_path
        self.annos_path = annos_path

    def __getitem__(self, item):  #for 
        img = self.imgs_path[item]
        anno = self.annos_path[item]

        pil_img = Image.open(img)
        img_tensor = transform(pil_img)

        anno_img = Image.open(anno)
        anno_img = anno_img.convert('L')  # Convert to grayscale
        anno_tensor = transform(anno_img)
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long)  # Remove singleton dimensions
        anno_tensor[anno_tensor > 0] = 1  # Binarize the mask
        return img_tensor, anno_tensor

    def __len__(self):
        return len(self.imgs_path)

# Create training and test datasets and dataloaders
train_ds = dataset(images, anno)
test_ds = dataset(test_images, test_anno)
train_dl = data.DataLoader(train_ds, batch_size=1, shuffle=True)
test_dl = data.DataLoader(test_ds, batch_size=1)
img_batch, anno_batch = next(iter(train_dl)) # Get a batch of images and annotations
#print(img_batch.shape)
#print(anno_batch.shape)

# Visualize a batch of images and corresponding annotations
img = img_batch[0].permute(1, 2, 0).numpy()  # Change the order of dimensions and convert to numpy
anno = anno_batch[0].numpy()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(anno)
plt.show()

# Define the decoder block 
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

# Define the main model
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

# Instantiate
model = OurModel(2)

# Move model to selected device
model.to(device)
    
# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
from torch.optim import lr_scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Store training and validation metrics
wtl_train_loss = []
wtl_test_loss = []
wtl_train_acc = []
wtl_test_acc = []

expand_epochs = 45
for epoch in range(expand_epochs):
    wt_epoch_loss, wt_epoch_acc, wt_test_epoch_loss, wt_test_epoch_acc = fit.fit(epoch, model, loss_fn, optimizer, train_dl, test_dl, exp_lr_scheduler)
    wtl_train_loss.append(wt_epoch_loss)
    wtl_test_loss.append(wt_epoch_acc)
    wtl_train_acc.append(wt_test_epoch_loss)
    wtl_test_acc.append(wt_test_epoch_acc)
    
    PATH = 'OurModel.pth'
    torch.save(model.state_dict(), PATH)

# save the model
PATH = 'OurModel.pth'
torch.save(model.state_dict(), PATH)
