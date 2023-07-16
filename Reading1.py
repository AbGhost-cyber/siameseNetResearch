import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

# Load the input image
input_image = Image.open("/Users/mac/Downloads/cat.jpeg")

# Define the desired output size
output_size = 100

# Define the CenterCrop transform
crop_transform = transforms.Compose([transforms.Resize(output_size), transforms.CenterCrop(100)])

# Apply the transform to the input image
output_image = crop_transform(input_image)

# Display the output image using matplotlib
plt.imshow(output_image)
plt.show()

if __name__ == '__main__':
    print()
