from google.colab import drive
drive.mount('/content/drive')

# Define the base directory for your dataset
base_dir = '/content/drive/MyDrive/MLSA BUP/drawings/Drawing'

# Install fastai from PyPI
!pip install -Uqq fastai

# Import required libraries
from fastai.vision import *
from fastai.metrics import accuracy
import numpy as np
from pathlib import Path  # Import Path from pathlib to work with file paths

# Define the base directory for your dataset
base_dir = '/content/drive/MyDrive/MLSA BUP/drawings'

# Define the paths for spiral and wave folders separately
spiral_path = Path(base_dir + '/spiral')
wave_path = Path(base_dir + '/wave')

print("Spiral data path:", spiral_path)
print("Wave data path:", wave_path)

# Define paths directly to training and testing subdirectories for spiral
spiral_train_path = spiral_path / 'training'
spiral_test_path = spiral_path / 'testing'

# Load and prepare spiral data using ImageDataLoaders with specific paths
spiral_data = ImageDataLoaders.from_folder(
    spiral_train_path.parent,
    train="training",
    valid="testing",
    item_tfms=Resize(128),
    batch_tfms=aug_transforms(),
    bs=8
)

# Display a sample batch from the spiral dataset
spiral_data.show_batch(figsize=(8,8))

# Define paths directly to training and testing subdirectories for wave
wave_train_path = wave_path / 'training'
wave_test_path = wave_path / 'testing'

# Load and prepare wave data using ImageDataLoaders with specific paths
wave_data = ImageDataLoaders.from_folder(
    wave_train_path.parent,
    train="training",
    valid="testing",
    item_tfms=Resize(128),
    batch_tfms=aug_transforms(),
    bs=8
)

# Display a sample batch from the wave dataset
wave_data.show_batch(figsize=(8,8))

# Initialize and train model on spiral data
spiral_learn = cnn_learner(spiral_data, resnet34, metrics=accuracy).to_fp16()
spiral_learn.fit_one_cycle(5)

# Plot losses using the recorder
spiral_learn.recorder.plot_loss()

# Initialize and train model on wave data
wave_learn = cnn_learner(wave_data, resnet34, metrics=accuracy).to_fp16()
wave_learn.fit_one_cycle(5)

# Plot losses using the recorder
wave_learn.recorder.plot_loss()

# Learning rate finder for spiral model
spiral_learn.lr_find()
spiral_learn.recorder.plot_lr_find()

# Learning rate finder for wave model
wave_learn.lr_find()
wave_learn.recorder.plot_lr_find()

# Unfreeze and train the spiral model with a specific learning rate
spiral_learn.unfreeze()
spiral_learn.fit_one_cycle(5, lr_max=1e-04)

# Unfreeze and train the wave model with a specific learning rate
wave_learn.unfreeze()
wave_learn.fit_one_cycle(5, lr_max=1e-04)

# Save spiral model
spiral_learn.save('spiral-stage-1-rn34')

# Interpret spiral model results
spiral_interp = ClassificationInterpretation.from_learner(spiral_learn)
spiral_interp.plot_top_losses(9, figsize=(12,10))  # Removed heatmap argument
spiral_interp.plot_confusion_matrix()

# Save wave model
wave_learn.save('wave-stage-1-rn34')

# Interpret wave model results
wave_interp = ClassificationInterpretation.from_learner(wave_learn)
wave_interp.plot_top_losses(9, figsize=(12,10))  # Removed heatmap argument
wave_interp.plot_confusion_matrix()

# Reload spiral data with 256x256 images
spiral_data = ImageDataLoaders.from_folder(
    spiral_train_path.parent,
    train="training",
    valid="testing",
    item_tfms=Resize(256),
    batch_tfms=aug_transforms(),
    bs=8
)

# Initialize and train the spiral model again
spiral_learn = cnn_learner(spiral_data, resnet34, metrics=accuracy).to_fp16()
spiral_learn.load('spiral-stage-1-rn34')  # Load previous weights
spiral_learn.fit_one_cycle(5)

# Reload wave data with 256x256 images
wave_data = ImageDataLoaders.from_folder(
    wave_train_path.parent,
    train="training",
    valid="testing",
    item_tfms=Resize(256),
    batch_tfms=aug_transforms(),
    bs=8
)

# Initialize and train the wave model again
wave_learn = cnn_learner(wave_data, resnet34, metrics=accuracy).to_fp16()
wave_learn.load('wave-stage-1-rn34')  # Load previous weights
wave_learn.fit_one_cycle(5)

# Final evaluation of spiral model
spiral_learn.show_results(figsize=(8,8))
spiral_interp = ClassificationInterpretation.from_learner(spiral_learn)
spiral_interp.plot_confusion_matrix()

# Save the final spiral model
spiral_learn.save('spiral-stage-2-rn34-256-final')
spiral_learn.export('spiral-parkinson-predictor.pkl')

# Final evaluation of wave model
wave_learn.show_results(figsize=(8,8))
wave_interp = ClassificationInterpretation.from_learner(wave_learn)
wave_interp.plot_confusion_matrix()

# Save the final wave model
wave_learn.save('wave-stage-2-rn34-256-final')
wave_learn.export('wave-parkinson-predictor.pkl')

from sklearn.metrics import classification_report, accuracy_score

# Get predictions and true labels for spiral model
spiral_preds, spiral_targets = spiral_learn.get_preds()
spiral_pred_labels = spiral_preds.argmax(dim=1)  # Convert probabilities to class labels

# Get predictions and true labels for wave model
wave_preds, wave_targets = wave_learn.get_preds()
wave_pred_labels = wave_preds.argmax(dim=1)  # Convert probabilities to class labels

# Generate classification report for spiral model
print("Spiral Model Classification Report:")
print(classification_report(spiral_targets, spiral_pred_labels, target_names=spiral_data.vocab))

# Calculate accuracy separately
spiral_accuracy = accuracy_score(spiral_targets, spiral_pred_labels)
print(f"Spiral Model Accuracy: {spiral_accuracy * 100:.2f}%")

# Generate classification report for wave model
print("Wave Model Classification Report:")
print(classification_report(wave_targets, wave_pred_labels, target_names=wave_data.vocab))

# Calculate accuracy separately
wave_accuracy = accuracy_score(wave_targets, wave_pred_labels)
print(f"Wave Model Accuracy: {wave_accuracy * 100:.2f}%")





