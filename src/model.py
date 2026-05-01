import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def create_model(num_classes=5):
    """
    Initializes a pre-trained ResNet-18 model, freezes its base layers, 
    and modifies the final classification layer for our specific dataset.
    """
    
    # 1. Load the pre-trained ResNet-18 model
    # We use DEFAULT weights which means it's pre-trained on ImageNet
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # 2. Freeze the base layers
    # This tells PyTorch NOT to update these weights during training, saving massive amounts of time
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Replace the final Fully Connected (fc) layer
    # ResNet-18's default final layer outputs 1000 classes. We change it to output our 5 classes.
    # By default, new layers created in PyTorch have requires_grad=True, so only this layer will train.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# --- Quick Test Block ---
if __name__ == "__main__":
    # Initialize the model
    net = create_model(num_classes=5)
    
    # Create a "dummy" batch of images (32 images, 3 channels, 224x224 pixels)
    # This simulates the exact output shape from your data_loader
    dummy_input = torch.randn(32, 3, 224, 224)
    
    # Pass the dummy data through the model
    output = net(dummy_input)
    
    print(f"Model successfully created!")
    print(f"Output Tensor Shape: {output.shape}") 
    # Expected output: [32, 5] (32 images, 5 probability scores each)   