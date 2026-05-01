import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir='data/raw/dataset-resized/dataset-resized', batch_size=32):
    """
    Loads the dataset, applies transformations, splits into train/val/test, 
    and returns PyTorch DataLoaders.
    """
    
    # 1. Define the Image Transformations
    # ResNet-18 expects images to be 224x224 and normalized in a very specific way
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),            # Resize all images to standard size
        transforms.RandomHorizontalFlip(p=0.5),   # Augmentation: flip 50% of images
        transforms.RandomRotation(10),            # Augmentation: slight rotation
        transforms.ToTensor(),                    # Convert image to PyTorch numerical tensor
        transforms.Normalize(                     # Standard ImageNet normalization
            mean=[0.485, 0.456, 0.406], 
            
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. Load the entire dataset from the folders
    # PyTorch's ImageFolder automatically uses the folder names (cardboard, glass, etc.) as labels
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    class_names = full_dataset.classes
    print(f"Found classes: {class_names}")
    print(f"Total images found: {len(full_dataset)}")

    # 3. Calculate split sizes (80% Train, 10% Validation, 10% Test)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size # The remainder

    # 4. Split the dataset randomly
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Ensures the split is the same every time
    )

    # 5. Create DataLoaders (this bundles the data into batches for training)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names

# --- Quick Test Block ---
if __name__ == "__main__":
    # If you run this script directly, it will test if everything works
    train_loader, val_loader, test_loader, classes = get_data_loaders()
    
    # Grab one batch of data to inspect
    images, labels = next(iter(train_loader))
    print(f"\nBatch Image Tensor Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")