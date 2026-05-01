import torch
import torch.nn as nn
import torch.optim as optim
import os

# Import the functions we built in the other files
from src.data_loader import get_data_loaders
from src.model import create_model

def train_model(epochs=5):
    # 1. Setup Device (Uses GPU if you have one, otherwise defaults to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load Data and Initialize Model
    print("Loading data and model...")
    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=32)
    
    # Create the model and send it to the CPU/GPU
    model = create_model(num_classes=len(classes)).to(device)

    # 3. Define the Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss() # Standard for classification
    # We ONLY optimize the parameters of the final 'fc' layer we added
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    print("Starting training loop...\n")
    # 4. The Main Training Loop
    for epoch in range(epochs):
        model.train() # Put model in training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Loop through batches of training data
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Reset the gradients (mistakes) from the last batch
            optimizer.zero_grad()

            # Forward pass: guess the labels
            outputs = model(images)
            
            # Calculate the loss (how wrong the guesses were)
            loss = criterion(outputs, labels)

            # Backward pass: calculate how to fix the weights
            loss.backward()
            
            # Update the weights
            optimizer.step()

            # Track accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # 5. Validation Phase (Check accuracy on unseen data)
        model.eval() # Put model in evaluation mode
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # Don't track gradients during validation
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 6. Save the trained model
    os.makedirs('models', exist_ok=True)
    save_path = 'models/waste_sorter.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model weights saved to {save_path}")

if __name__ == "__main__":
    train_model(epochs=5)