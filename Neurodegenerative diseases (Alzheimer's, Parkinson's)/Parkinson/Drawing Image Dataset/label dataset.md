To assign labels to images based on their folder structure, you can use a script in Python. This script will load each image and assign a label based on whether it’s in the "healthy" or "parkinson" folder, creating a structured dataset with labels.

Here’s a step-by-step guide using Python with `os` for folder navigation and `PIL` (from `Pillow`) for image handling:

---

### Step 1: Import Required Libraries

```python
import os
from PIL import Image
import pandas as pd
```

### Step 2: Set Up Directory Paths

Adjust the `base_dir` path to the location of your main `Drawing` directory.

```python
# Base directory containing spiral and wave images
base_dir = "path/to/Drawing"
output_data = []  # This will store image paths and labels
```

### Step 3: Label Assignment Logic

Define a loop that navigates through each folder in the `Drawing` structure and assigns labels based on folder names.

```python
# Iterate through each drawing type (spiral and wave)
for drawing_type in ['spiral', 'wave']:
    drawing_dir = os.path.join(base_dir, drawing_type)
    
    # Iterate through each dataset split (training and testing)
    for split in ['training', 'testing']:
        split_dir = os.path.join(drawing_dir, split)
        
        # Iterate through health condition folders
        for condition in ['healthy', 'parkinson']:
            condition_dir = os.path.join(split_dir, condition)
            
            # Label 0 for healthy, 1 for parkinson
            label = 0 if condition == 'healthy' else 1
            
            # Iterate through each image in the condition folder
            for image_name in os.listdir(condition_dir):
                image_path = os.path.join(condition_dir, image_name)
                
                # Append image path and label to the list
                output_data.append({
                    "image_path": image_path,
                    "label": label,
                    "drawing_type": drawing_type,
                    "split": split
                })
```

### Step 4: Convert to a DataFrame (Optional)

If you’d like to visualize or save the dataset structure, convert `output_data` to a DataFrame:

```python
# Create a DataFrame to organize the data
df = pd.DataFrame(output_data)
print(df.head())  # Display the first few rows of the DataFrame

# Save to a CSV if needed
df.to_csv("drawing_dataset_labels.csv", index=False)
```

### Explanation of the Output

- The DataFrame `df` will contain each image’s path, label (`0` for healthy, `1` for Parkinson’s), drawing type (`spiral` or `wave`), and dataset split (`training` or `testing`).
- Saving this structure to a CSV (`drawing_dataset_labels.csv`) will help you easily access and use the data with consistent labels for training or validation in ML models.

---

### Next Steps

You can now use this labeled data to load images with their corresponding labels in a machine learning pipeline or use it to train a federated learning model where images remain on local devices but are labeled consistently. 

Let me know if you'd like help with further steps, such as image loading for model training!
