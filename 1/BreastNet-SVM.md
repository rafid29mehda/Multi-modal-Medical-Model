To implement the **BreastNet-SVM** model for the breast cancer detection project, I'll walk we through the code setup in parts. This model combines features from deep learning (AlexNet) and a Support Vector Machine (SVM) for classification. We'll use **AlexNet** to extract features and **SVM** for final classification. Since we're working with mammography PNG images, this setup will focus on feature extraction from images and training the SVM.

### **Step 1: Install Necessary Libraries**

```python
# Step 1: Install required libraries
!pip install tensorflow keras scikit-learn opencv-python

import tensorflow as tf
from tensorflow.keras.applications import AlexNet
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dense, Dropout
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
```

### **Step 2: Load and Preprocess Images**

We’ll use `cv2` (OpenCV) for image loading and resizing since BreastNet-SVM typically uses small images like **32x32**.

```python
# Step 2: Load and preprocess images
def load_and_preprocess_image(img_path, target_size=(32, 32)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # Normalize the image
    return img

# Paths for benign and malignant masses
benign_path = '/content/INbreast+MIAS+DDSM/INbreast+MIAS+DDSM Dataset/Benign Masses'
malignant_path = '/content/INbreast+MIAS+DDSM/INbreast+MIAS+DDSM Dataset/Malignant Masses'

def load_images_from_directory(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = load_and_preprocess_image(img_path)
        images.append(img)
        labels.append(label)  # 0 for benign, 1 for malignant
    return np.array(images), np.array(labels)

# Load benign and malignant images
benign_images, benign_labels = load_images_from_directory(benign_path, 0)
malignant_images, malignant_labels = load_images_from_directory(malignant_path, 1)

# Combine the data
X = np.concatenate((benign_images, malignant_images), axis=0)
y = np.concatenate((benign_labels, malignant_labels), axis=0)
```

### **Step 3: Feature Extraction using AlexNet**

We’ll use a pre-trained **AlexNet** model to extract features, which will then be fed into the SVM classifier.

```python
# Step 3: Feature extraction using AlexNet
base_model = AlexNet(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Add a global spatial average pooling layer
x = Flatten()(base_model.output)
x = Dropout(0.5)(x)  # Add dropout for regularization
model = Model(inputs=base_model.input, outputs=x)

# Extract features for all images
features = model.predict(X)

# Reshape the features for SVM input
features_flattened = features.reshape(features.shape[0], -1)
```

### **Step 4: Train the SVM Classifier**

Once the features are extracted, we’ll train an SVM on these features for the binary classification task (benign vs. malignant).

```python
# Step 4: Train the SVM classifier
from sklearn.model_selection import train_test_split

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_flattened, y, test_size=0.2, random_state=42)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = svm_classifier.predict(X_val)

# Evaluate the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
```

### **Step 5: Save and Load the Model**

we can save the trained SVM model for later use and load it when needed.

```python
import joblib

# Save the model
joblib.dump(svm_classifier, 'breastnet_svm_model.pkl')

# Load the model
svm_classifier = joblib.load('breastnet_svm_model.pkl')
```

### **Step 6: Prediction on New Images**

Finally, we can load new images, preprocess them, extract features using **AlexNet**, and then classify them using the trained SVM model.

```python
# Step 6: Predict on new images
def predict_image(img_path):
    img = load_and_preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)
    
    # Extract features
    features = model.predict(img)
    features_flattened = features.reshape(1, -1)
    
    # Predict using the SVM classifier
    prediction = svm_classifier.predict(features_flattened)
    
    if prediction == 0:
        print("Predicted: Benign")
    else:
        print("Predicted: Malignant")

# Test with a new image
predict_image('/path_to_new_image.png')
```

### **Summary of Steps**:
1. **Image Preprocessing**: The dataset is loaded, and images are resized to 32x32.
2. **Feature Extraction**: A pre-trained **AlexNet** model is used to extract features from the images.
3. **SVM Classifier**: The extracted features are passed to a **Support Vector Machine** classifier for training.
4. **Model Saving and Loading**: The trained model can be saved and loaded for later predictions.
5. **Prediction**: Classifies new images as benign or malignant based on the trained model.

This approach effectively implements **BreastNet-SVM** for mammography image classification. Let me know if we have any questions!
