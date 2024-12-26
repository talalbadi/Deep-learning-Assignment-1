import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from EDF import *

def load_image_as_grayscale(path):

    img = Image.open(path).convert("L")
    img_array = np.array(img, dtype=np.float32)
    return img_array

def apply_kernel_to_image(image_path, kernel):

    img_gray = load_image_as_grayscale(image_path)
    conv_node = Conv(kernel, img_gray)
    conv_node.forward()
    filtered_img = conv_node.value
    return img_gray, filtered_img

if __name__ == "__main__":
    example_kernel = np.array([
        [1, 0, -1],
        [3,  -1, -1],
        [1, 0, -1]
    ], dtype=np.float32)
    
    image_path = "C:/dev/EE569/Assignment1/part-c/task2/your_image.jpg"
    original, filtered = apply_kernel_to_image(image_path, example_kernel)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original Grayscale")
    plt.imshow(original, cmap="gray")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("Filtered")
    plt.imshow(filtered, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
