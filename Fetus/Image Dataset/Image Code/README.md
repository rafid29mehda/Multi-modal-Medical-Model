This code demonstrates how to train a **Convolutional Neural Network (CNN)** for classifying fetal health based on **2D fetal ultrasound images**. Here’s an easy-to-understand breakdown of the code, step by step:

---

### **1. Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
data_dir = '/content/drive/MyDrive/MLSA BUP/Classification'
```
- **Purpose**: Mount Google Drive to access the dataset stored in your Drive.
- **`data_dir`**: Sets the path to the directory containing the image dataset and labels.

---

### **2. Load the Labels**
```python
csv_path = f'{data_dir}/image_label.csv'
dataset = pd.read_csv(csv_path)
print(dataset.head())
print("Class Distribution:\n", dataset['Plane'].value_counts())
```
- **Purpose**: Reads the CSV file containing image file names and their corresponding class labels.
- **`dataset.head()`**: Displays the first few rows of the dataset.
- **Class distribution**: Prints the number of images per class to check for imbalance in the dataset.

---

### **3. Load and Preprocess Images**
```python
image_dir = f'{data_dir}/images'
img_size = (256, 256)
label_mapping = {label: idx for idx, label in enumerate(dataset['Plane'].unique())}
dataset['Label'] = dataset['Plane'].map(label_mapping)
```
- **Image Directory**: Specifies the folder where images are stored.
- **Image Size**: Sets the size to resize images (256x256 pixels).
- **Label Mapping**: Converts class names (e.g., `Normal`, `Pathological`) to integer labels for easier processing. E.g., `Normal → 0`, `Pathological → 1`.

#### Function: Load Images
```python
def load_data(dataset, image_dir):
    images = []
    labels = []
    for _, row in dataset.iterrows():
        image_path = os.path.join(image_dir, f"{row['Image_name']}.png")
        if os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, img_size)
            images.append(img_resized)
            labels.append(row['Label'])
        else:
            print(f"Image not found: {image_path}")
    return np.array(images), np.array(labels)
```
- **Purpose**: Loads all images from the directory:
  - Converts images to grayscale.
  - Resizes them to `256x256` pixels.
  - Normalizes the pixel values (dividing by 255 is done later).
  - Maps the image to its corresponding label.

---

### **4. Normalize and Split Data**
```python
images = images[..., np.newaxis] / 255.0
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
```
- **Normalize**: Rescales pixel values to be between 0 and 1.
- **Split**: Divides the dataset into:
  - **Training set (80%)**: For training the CNN.
  - **Validation set (20%)**: For evaluating the model during training.
- **Stratify**: Ensures the same proportion of classes in both training and validation sets.

---

### **5. Define the CNN**
```python
def create_model(input_shape=(256, 256, 1), num_classes=num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```
- **Purpose**: Defines a CNN with:
  - **Conv2D Layers**: Extract features from images.
  - **MaxPooling**: Reduces the spatial size of features to reduce computation.
  - **Flatten**: Converts the 2D feature map into a 1D vector.
  - **Dense Layer**: Fully connected layer for classification.
  - **Dropout**: Prevents overfitting.
  - **Softmax Activation**: Outputs probabilities for each class.

---

### **6. Compile and Summarize the Model**
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```
- **Optimizer**: Uses Adam with a reduced learning rate for better convergence.
- **Loss Function**: `sparse_categorical_crossentropy` for multi-class classification.
- **Model Summary**: Prints the architecture of the CNN.

---

### **7. Handle Class Imbalance**
```python
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(labels), y=labels
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
```
- **Purpose**: Adjusts the loss calculation to account for imbalanced classes by assigning higher weights to underrepresented classes.

---

### **8. Define Callbacks**
```python
checkpoint_path = '/content/drive/MyDrive/MLSA BUP/best_model.keras'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]
```
- **Model Checkpoint**: Saves the best model based on validation accuracy.
- **Early Stopping**: Stops training if validation accuracy doesn't improve for 10 epochs.
- **ReduceLROnPlateau**: Reduces the learning rate when validation loss stops improving.

---

### **9. Data Augmentation**
```python
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)
train_generator = datagen.flow(X_train, y_train, batch_size=32)
```
- **Purpose**: Augments the training data to make the model more robust by:
  - Rotating, shifting, zooming, and flipping images.
- **Batch Size**: Specifies the number of images in each batch during training.

---

### **10. Train the Model**
```python
history = model.fit(train_generator, epochs=20, validation_data=(X_val, y_val),
                    class_weight=class_weight_dict, callbacks=callbacks)
```
- **Purpose**: Trains the model with data augmentation, class weights, and callbacks.

---

### **11. Evaluate and Plot Results**
```python
val_loss, val_accuracy = model.evaluate(X_val, y_val)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.show()
```
- **Validation Accuracy**: Evaluates the model’s performance on the validation set.
- **Plots**: Shows training and validation accuracy/loss over epochs to assess overfitting.

---

### **12. Load and Predict Test Images**
```python
X_test, test_image_names = load_test_images(test_image_dir)
predictions = model.predict(X_test)
predicted_labels = [list(label_mapping.keys())[np.argmax(pred)] for pred in predictions]
```
- **Load Test Images**: Preprocesses external test images.
- **Predict**: Predicts class probabilities for each image and decodes them into class labels.

---

### **13. Display Predictions**
```python
for i in range(len(X_test)):
    plt.imshow(X_test[i].reshape(256, 256), cmap='gray')
    plt.title(f"Predicted: {predicted_labels[i]}")
    plt.axis('off')
    plt.show()
```
- **Purpose**: Displays test images with their predicted labels for visual inspection.

---

### **14. Confidence Scores**
```python
for i, pred in enumerate(predictions):
    confidence = np.max(pred)
    predicted_label = predicted_labels[i]
    print(f"Image: {test_image_names[i]}, Predicted: {predicted_label}, Confidence: {confidence:.2f}")
```
- **Purpose**: Displays the confidence score (probability) for each prediction.

---

Let me know if you'd like to dive deeper into any section or improve specific parts!
