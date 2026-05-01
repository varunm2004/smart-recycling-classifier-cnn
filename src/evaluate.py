import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import our custom functions
from src.data_loader import get_data_loaders
from src.model import create_model

def evaluate_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Load the data (we only need the test_loader and classes here)
    print("Loading test data...")
    _, _, test_loader, classes = get_data_loaders(batch_size=32)
    
    # 3. Initialize the model and load the trained weights
    print("Loading trained model weights...")
    model = create_model(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load('models/waste_sorter.pth', map_location=device))
    model.eval() # Put model in evaluation mode (turns off dropout/batchnorm updates)

    all_preds = []
    all_labels = []

    print("Running test dataset through the model...\n")
    # 4. Run Inference on unseen data
    with torch.no_grad(): # Disable gradient calculation to save memory and speed up
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get the highest probability prediction
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and true labels for metric calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Print the Classification Report
    print("--- Final Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 6. Generate and Save a Confusion Matrix Visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted Material')
    plt.ylabel('Actual Material')
    plt.title('Waste Sorter - Confusion Matrix')
    
    # Save the visualization to the main folder
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    print("\nVisualization saved as 'confusion_matrix.png' in your project root folder!")

if __name__ == "__main__":
    evaluate_model()