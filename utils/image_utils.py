import os
import shutil
from shutil import move


def select_images(source_folder, destination_folder, num_images_per_index=12):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over the files in the source folder
    for filename in os.listdir(source_folder):
        if not filename.__contains__("png"):
            continue
        # Split the filename into parts
        parts = filename.split('_')

        # Extract the index and sub-index
        index = int(parts[1])
        sub_index = int(parts[2].split('.')[0])

        # Check if the current index and sub-index meet the criteria
        if sub_index <= num_images_per_index:
            # Construct the source and destination paths
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            # Copy the image to the destination folder
            shutil.copyfile(source_path, destination_path)


# Specify the source and destination folders and the number of images per index
full_org_folder = '/Users/mac/PycharmProjects/signetTest/data/BHSig260/full_org'
full_forg_folder = '/Users/mac/PycharmProjects/signetTest/data/BHSig260/full_forg'
new_full_org_folder = '/Users/mac/PycharmProjects/signetTest/data/BHSig260/full_new_org'
new_full_forg_folder = '/Users/mac/PycharmProjects/signetTest/data/BHSig260/full_new_forg'

# Select and copy the desired images for genuine signatures
select_images(full_org_folder, new_full_org_folder)

# Select and copy the desired images for forged signatures
select_images(full_forg_folder, new_full_forg_folder)


def organize_images_into_subfolders(root_folder):
    for filename in os.listdir(root_folder):
        if filename.endswith(".png"):
            parts = filename.split("_")
            original_index = int(parts[1])
            folder_name = f"s{original_index}"
            folder_path = os.path.join(root_folder, folder_name)

            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(root_folder, filename)
            new_file_path = os.path.join(folder_path, filename)
            move(file_path, new_file_path)


original_signatures_root = '/Users/mac/PycharmProjects/signetTest/data/BHSig260/full_new_org'
forg_signatures_root = '/Users/mac/PycharmProjects/signetTest/data/BHSig260/full_new_forg'
# Usage example:
organize_images_into_subfolders(original_signatures_root)
organize_images_into_subfolders(forg_signatures_root)
print("done")

if __name__ == '__main__':
    print()
