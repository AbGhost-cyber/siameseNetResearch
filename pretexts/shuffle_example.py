import random

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image


def extract_patches(original):
    image_dimension = 224
    num_division = 3
    patch_size = image_dimension // num_division
    crop_64 = transforms.RandomCrop(64)  # Crop patches of size 64x64
    color_transform = transforms.Grayscale()  # Convert the patches to grayscale
    final_transform = transforms.Compose([
        crop_64,
        color_transform,
        transforms.ToTensor()  # Convert the patches to tensors
    ])

    # Crop a sample from the original image
    sample = transforms.RandomCrop(image_dimension + 1)(original)

    # Define the cropped areas based on the patch size
    cropped_areas = []
    for i in range(num_division):
        for j in range(num_division):
            # Calculate the coordinates for cropping each patch
            cropped_area = (i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size)
            cropped_areas.append(cropped_area)

    # Crop the patches from the sample
    samples = [sample.crop(cropped_area) for cropped_area in cropped_areas]

    # Apply the final transformations to each patch
    samples = [final_transform(patch) for patch in samples]

    # Shuffle the patches randomly
    random.shuffle(samples)

    return samples


# Example usage
image = Image.open("/Users/mac/Downloads/samia.jpeg")
extracted_patches = extract_patches(image)
print(len(extracted_patches))
fig, axes = plt.subplots(3, 3, figsize=(7, 7))
for i, ax in enumerate(axes.flat):
    patch = extracted_patches[i].permute(1, 2, 0)  # Transpose the dimensions for plotting
    print(f"patch shape {patch.shape}")
    ax.imshow(patch, cmap='gray')
    ax.axis('off')

plt.show()
