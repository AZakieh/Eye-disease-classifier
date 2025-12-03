import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Configuration Constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 2  # 0=Healthy, 1=Disease (Change this if you add more diseases)
NUM_EPOCHS = 2

# Ignore warnings to keep console clean
import warnings
warnings.filterwarnings("ignore")



class EyeLandmarksDataset(Dataset):
    """ 
    Custom Dataset class. 
    Currently generates 'Mock' random data for testing the pipeline.
    In the future, this will load real images from folders/CSVs.
    """

    def __init__(self, csv_file=None, root_dir=None, transform=None):
        self.length = 100  # Pretend we have 100 images
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # Returns the total size of the dataset
        return self.length
    
    def __getitem__(self, idx):
        """
        Fetches one single item (image + label) when asked.
        """
        # Generate a fake eye image (Random Noise)
        # Shape: (3 Color Channels, 224 Height, 224 Width)
        fake_image = torch.randn(3, 224, 224)

        # Generate a fake label (0 or 1)
        fake_label = torch.randint(0, NUM_CLASSES, (1,)).item()

        return fake_image, fake_label


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
    
    #Setup Data
    dataset = EyeLandmarksDataset()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setup Model
    model = EyeResNet()

    # Run Training
    train_engine(model, train_loader)