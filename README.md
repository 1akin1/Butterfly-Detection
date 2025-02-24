# Butterfly Detection using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to detect whether an image contains a butterfly. The model is trained on a dataset of labeled images and can predict the presence of a butterfly with a certain confidence level.

## Dataset
The dataset consists of labeled images stored in a CSV file (`Training_set.csv`) with corresponding images in a specified directory. The labels indicate whether an image contains a butterfly or not.

## Dependencies
Ensure you have the following dependencies installed:

```bash
pip install tensorflow keras numpy pandas matplotlib pillow scikit-learn
```

## Model Architecture
The CNN model consists of:
- 3 convolutional layers with ReLU activation
- MaxPooling layers to reduce dimensionality
- A fully connected dense layer with ReLU activation
- A final output layer with a sigmoid activation for binary classification

## Training the Model
To train the model, run the script:

```python
cnn.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=10)
```

After training, the model achieves high accuracy on the validation dataset and is saved for later use:

```python
cnn.save('dataset/cnn_binary_detection.keras')
```

## Testing the Model
To test an image for butterfly detection, use the following script:

```python
from PIL import Image
import numpy as np

# Load an image to test
test_image_path = 'path/to/image.png'
test_image = Image.open(test_image_path).convert("RGB")
test_image = test_image.resize((64, 64))
test_image = np.array(test_image) / 255.0
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

# Predict whether the image contains a butterfly
prediction = cnn.predict(test_image)

if prediction > 0.5:
    confidence = prediction[0][0] * 100  # Convert probability to percentage
    print(f"The image contains a butterfly with {confidence:.2f}% confidence.")
else:
    confidence = (1 - prediction[0][0]) * 100  # Convert probability to percentage
    print(f"The image does not contain a butterfly with {confidence:.2f}% confidence.")
```

## Results
The model achieves:
- Training Accuracy: ~97.68%
- Validation Accuracy: ~94.88%

## Future Improvements
- Enhance dataset with more diverse butterfly images.
- Experiment with deeper architectures and data augmentation.
- Implement a real-time detection system using a webcam.


## Author
Developed by 1akin1.

