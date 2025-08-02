### **Step 1: Install YOLOv5 and Required Libraries**

```python
# Step 1: Clone YOLOv5 repository and install required dependencies
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
```

### **Step 2: Organize Your Dataset**

To use YOLO, your dataset must be structured in the following format:

```
/content/INbreast+MIAS+DDSM Dataset/
    train/
        images/
            img1.png
            img2.png
            ...
        labels/
            img1.txt
            img2.txt
            ...
    val/
        images/
        labels/
```

Each `.txt` file must contain the bounding box coordinates and class labels in YOLO format.

### **Step 3: Label Images**

Use a tool like **LabelImg** to annotate your images and generate bounding box coordinates for benign and malignant regions. The annotations should be saved in YOLO format.

- **Benign = 0**
- **Malignant = 1**

For example, the content of `img1.txt` might look like this:
```
0 0.5 0.5 0.2 0.3  # (Class 0 = benign, x_center, y_center, width, height)
```

### **Step 4: Create a Custom Data Configuration File**

Create a custom YAML file, `dataset.yaml`, to define your dataset paths and classes.

```yaml
train: /content/INbreast+MIAS+DDSM Dataset/train/images  # path to training images
val: /content/INbreast+MIAS+DDSM Dataset/val/images  # path to validation images

nc: 2  # number of classes (benign, malignant)
names: ['Benign', 'Malignant']  # class names
```

### **Step 5: Train YOLOv5 on Your Dataset**

```python
# Step 5: Train YOLOv5
!python train.py --img 640 --batch 16 --epochs 50 --data /content/dataset.yaml --weights yolov5s.pt --cache
```

- `--img 640`: Image size for training
- `--batch 16`: Batch size
- `--epochs 50`: Number of training epochs
- `--weights yolov5s.pt`: Pretrained YOLOv5 model
- `--data`: Path to the dataset YAML file

### **Step 6: Evaluate the Model**

Once training is complete, evaluate the model's performance on the validation set:

```python
# Step 6: Evaluate model performance
!python val.py --data /content/dataset.yaml --weights runs/train/exp/weights/best.pt --img 640
```

### **Step 7: Inference on New Images**

To run inference on new mammography images:

```python
# Step 7: Inference on new images
!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source /path_to_new_images/
```

- `--weights`: Path to the trained model weights.
- `--img`: Image size.
- `--conf`: Confidence threshold.
- `--source`: Path to the new images folder.

### **Step 8: Visualize the Results**

After inference, the results will be saved in the `runs/detect/exp` directory. You can visualize the results:

```python
import matplotlib.pyplot as plt
import cv2

# Load and display result image
result_img = cv2.imread('/content/yolov5/runs/detect/exp/image.jpg')
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.show()
```

### **Step 9: Save and Load the Model**

You can save and load the trained YOLOv5 model for future use:

```python
# Save the model
!cp runs/train/exp/weights/best.pt /content/drive/MyDrive/your_model_name.pt

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='your_model_name.pt')
```

### **Summary of Steps**:
1. **Install YOLOv5**: Clone the repository and install dependencies.
2. **Prepare Dataset**: Organize images and labels in YOLO format.
3. **Label Images**: Use LabelImg for bounding box annotations.
4. **Train Model**: Train the YOLOv5 model on your dataset.
5. **Evaluate Model**: Evaluate model performance on the validation set.
6. **Inference**: Run inference on new images and visualize the results.
