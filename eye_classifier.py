import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
from PIL import Image

# Configuration Constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 2  # 0=Healthy, 1=Disease (Change this to add more diseases)
NUM_EPOCHS = 2

# Ignore warnings to keep console clean
import warnings
warnings.filterwarnings("ignore")



class EyeLandmarksDataset(Dataset):
    """ 
    Standard Dataset class for loading images from a directory structure.
    Expected structure:
    root_dir/
       healthy/
          img1.jpg
       disease/
          img2.jpg
    """

    def __init__(self, csv_file=None, root_dir=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        #if root_dir exists, use ImageFolder to list all the files automatically
        if os.path.exists(root_dir):
            self.data = datasets.ImageFolder(root_dir, transform=transform)
        else:
            self.data = []
            print(f"{root_dir} not found. Dataset is empty.")

    def __len__(self):
        return len(self.data) if self.data else 0 
    
    def __getitem__(self, idx):
       return self.data[idx]


class EyeResNet(nn.Module):
    """
    Custom ResNet18 model for eye disease classification.
    """
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights="DEFAULT")
        
        self.model.fc = nn.Linear(in_features=512, out_features=NUM_CLASSES)
    
    def forward(self, x):
        # Define how data flows through the network
        return self.model(x)



def load_model_for_inference():
    """
    Creates a blank model and loads the saved 'Memory' (weights) from disk.
    """
    model = EyeResNet()
    model.load_state_dict(torch.load("eye_model.pth", weights_only=True))
    
    model.eval()
    return model

def predict_image(model, image_path):
    """
    Takes a single image file path and returns the predicted class index.
    """
    # Load Image
    im = Image.open(image_path)
    im = im.convert("RGB") # Ensure it has 3 channels (no transparency)

    # Preprocess (Must match the training transforms!)
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    im = transform_pipeline(im)

    # Add Batch Dimension (The "Fake Batch")
    # Model expects (Batch, Channel, Height, Width) -> (1, 3, 224, 224)
    im = torch.unsqueeze(im, 0)

    with torch.no_grad():
        guess = model(im)

    result = torch.argmax(guess, dim=1)
    
    return result.item()


def train_engine(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting Training Loop...")

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            images_batch, labels_batch = data

         
            optimizer.zero_grad()

            # Forward Pass (Guess)
            outputs = model(images_batch)

            # Calculate Error (Loss)
            loss = criterion(outputs, labels_batch)

            # Backward Pass (Blame)
            # Calculate which weights contributed to the error
            loss.backward()

            # Optimizer Step (Update)
           
            optimizer.step()

            # Print stats
            running_loss += loss.item()
            if i % 1 == 0: # Print every batch (since we have small mock data)
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss:.3f}')
                running_loss = 0.0
                
    print("Finished Training.")
    
    # Save the brain to disk
    torch.save(model.state_dict(), "eye_model.pth")
    print("Model saved to 'eye_model.pth'")



if __name__ == "__main__":
    
    data_path = "./data"

    #Setup Data
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = EyeLandmarksDataset(root_dir=data_path, transform=train_transforms)

    if len(dataset) > 0:
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        # Setup Model
        model = EyeResNet()

        # Run Training
        train_engine(model, train_loader)
    else:
        #saves an empty model so inference doesnt fail
        model = EyeResNet()
        torch.save(model.state_dict(), "eye_model.pth")