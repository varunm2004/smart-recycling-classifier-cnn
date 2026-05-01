# Automated Waste Sorter (CNN) 

An end-to-end computer vision pipeline that utilizes Transfer Learning (ResNet-18) to autonomously categorize recyclable materials and waste.

This project aims to improve recycling efficiency by accurately classifying images of trash into six distinct categories: Cardboard, Glass, Metal, Paper, Plastic, and General Trash.

 ## Methodology & Architecture

Instead of training a Convolutional Neural Network (CNN) from scratch, this project leverages Transfer Learning.

    Base Model: A pre-trained ResNet-18 architecture (trained on ImageNet) acts as the feature extractor, identifying fundamental shapes, edges, and textures.

    Fine-Tuning: The base convolutional layers are frozen to retain their learned weights. The final fully connected classification layer is replaced and dynamically trained to output predictions specifically for the waste categories.

    Data Augmentation: To prevent overfitting and simulate real-world environmental variations, the training pipeline applies dynamic random cropping, horizontal flipping, and rotational shifts to the dataset.


 ## Installation & Setup

1. Clone the repository
Bash

git clone https://github.com/varunm2004/smart-recycling-classifier-cnn.git
cd smart-recycling-classifier-cnn

2. Set up a virtual environment
Bash

python -m venv venv
source venv/Scripts/activate  # On Windows Git Bash

3. Install dependencies
Bash

pip install -r requirements.txt
pip install huggingface_hub

4. Download and Prepare the Dataset
The model is trained on the TrashNet dataset. Use the Hugging Face CLI to download the data directly into your workspace:
Bash

hf download garythung/trashnet --repo-type dataset --local-dir data/raw

5. Clean the Data Directory

    Extract the dataset-resized.zip archive into the data/raw/ directory.

    Important: Delete the .cache and __MACOSX folders generated during the download and extraction processes, as these hidden files will cause PyTorch's ImageFolder to crash.

    Ensure your data_loader.py points to the final extracted image directory (e.g., data/raw/dataset-resized/dataset-resized).

## Usage

### Training the Model

To start the training loop from scratch, run:
Bash

python -m src.train

This will process the data, train the final layer using the Adam optimizer and Cross-Entropy Loss, and save the highest-performing weights to models/waste_sorter.pth.

### Evaluating the Model
To test the model's accuracy on unseen data and generate a performance breakdown:
Bash

python -m src.evaluate

This will output a detailed classification report (Precision, Recall, F1-Score) and save a confusion_matrix.png visualization to the root directory.
📊 Results

The model leverages a pre-trained ResNet-18 network fine-tuned over 5 epochs. Based on internal testing splits, it achieves rapid convergence, establishing baseline accuracies above 78% on the training set and ~74% on the validation set with standard data augmentation techniques applied.

Authors: Varun Menon and Brendan McDonnell
