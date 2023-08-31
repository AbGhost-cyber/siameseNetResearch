import torch
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from PIL import Image

import torch
from torchvision import transforms


def shuffle_patches(image, patch_size):
    transform = transforms.Compose([
        transforms.RandomCrop(patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    print(f"original image length: {len(image)}")
    print(f"original image shape: {image.shape}")

    patches = []
    for _ in range(len(image)):
        patch = transform(image)
        patches.append(patch)
    shuffled_image = torch.cat(patches, dim=0)
    # shuffled_image = shuffled_image.view(-1, 3, patch_size, patch_size)
    print(f"original shuffled image length: {len(shuffled_image)}")
    print(f"original shuffled image shape: {shuffled_image.shape}")
    return shuffled_image


original_img = Image.open("/Users/mac/Downloads/samia.jpeg")
original_img_tensor = transforms.ToTensor()(original_img)
patch_size = 32
shuffled_img_tensor = shuffle_patches(original_img_tensor, patch_size)
num_patches = shuffled_img_tensor.shape[0]
patch_index = 0
selected_patch_tensor = shuffled_img_tensor[patch_index]
selected_patch_array = selected_patch_tensor.numpy().transpose(1, 2, 0)
print(selected_patch_array.size())
plt.imshow(selected_patch_array)
plt.axis('off')
plt.show()

# num_subplots = min(num_patches, 9)  # Set the maximum number of subplots to 9
# shuffled_img = shuffled_img_tensor.view(num_patches // 3, 3, patch_size, patch_size)
# print(f"size {shuffled_img.size()}")
# shuffled_img = shuffled_img.permute(0, 2, 3, 1).numpy()
#
# # Visualize the shuffled image
# fig, axs = plt.subplots(3, num_patches // 3, figsize=(num_patches // 3, 3))
# for i, ax in enumerate(axs.flat):
#     if i < 3:
#         ax.imshow(shuffled_img[i])
#     ax.axis('off')
#
# plt.show()

if __name__ == '__main__':
    print()
