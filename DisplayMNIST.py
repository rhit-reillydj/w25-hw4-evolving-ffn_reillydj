import numpy as np
import struct

def load_mnist_images(file_path, num_images_to_read):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number {magic} for images file."
        # Read only the required number of images
        images = np.frombuffer(f.read(num_images_to_read * rows * cols), dtype=np.uint8)
        # Reshape to NumPy array (num_images, 784)
        images = images.reshape(num_images_to_read, rows * cols).astype(np.float32)
        # Normalize pixel values to range 0-1
        images /= 255.0
    return images

def load_mnist_labels(file_path, num_labels_to_read):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number {magic} for labels file."
        # Read only the required number of labels
        labels = np.frombuffer(f.read(num_labels_to_read), dtype=np.uint8)
    return labels

# Specify number of images to read
num_images_to_read = 1

# Load and normalize images, and load labels
images_path = 'TrainingData/train-images-idx3-ubyte'
labels_path = 'TrainingData/train-labels-idx1-ubyte'

images = load_mnist_images(images_path, num_images_to_read)
labels = load_mnist_labels(labels_path, num_images_to_read)

# Verify normalized images
print(f"Images shape: {images.shape}")  # (10, 784)
print(f"Labels: {labels.tolist()}")     # Print labels as a Python list
print(f"First image pixel values (normalized): {images[0][:10]}")  # First 10 pixels of the first image

# Display the first image
import matplotlib.pyplot as plt

plt.imshow(images[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {labels[0]}")
plt.show()