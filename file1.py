import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()

class EyeLandmarksDataset(Dataset):
    """ Eye landmarks dataset """

    def __init__(self, csv_file=None, root_dir=None, tarnsform=None):
        """
        Arg:
            csv_file (string): path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = 100 #pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # for now returning that we have 100 eye image scans
        return self.landmarks_frame #len(self.landmarks_frame)#
    
    def __getitem__(self, idx):
        # creating a fake eye image
        # torch.randn generates numbers in a normal distribution

        fake_image = torch.randn(3, 224, 224)

        fake_label = torch.randint(0, 2, (1,)).item()

        return fake_image, fake_label
        

class EyeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights="DEFAULT")
        self.model.fc = nn.Linear(in_features=512, out_features=2)
        
    
    def forward(self, x):
        result = self.model(x)
        return result


def train_engine(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images_batch, labels_batch = data

            optimizer.zero_grad()

            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1:.3f}')
                running_loss = 0.0  
    print("finished training")
    torch.save(model.state_dict(), "eye_model.pth")

def load_model_for_inference():
    model = EyeResNet()
    model.load_state_dict(torch.load("eye_model.pth"), weights_only=True)
    model.eval()
    return model

def predict_image(model, image_path):
    im = Image.open(image_path)
    im = im.convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    im = transform(im)
    im = torch.unsqueeze(im, 0)
    with torch.no_grad():
        guess = model(im)
    result = torch.argmax(guess, dim=1)
    return result.item()







dataset = EyeLandmarksDataset()



train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
model = EyeResNet()
train_engine(model, train_loader)




