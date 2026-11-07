import os
import shutil


def move_images(src_folder, dst_folder, file_extensions=('.png', '.jpg', '.jpeg')):
    """
    Move all images from subfolders in the src_folder to dst_folder.

    Args:
    - src_folder: str, source directory containing subfolders with images.
    - dst_folder: str, destination directory to move all images.
    - file_extensions: tuple, allowed image file extensions.
    """
    # Ensure destination directory exists
    os.makedirs(dst_folder, exist_ok=True)

    # Walk through the source folder
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            # Check if file is an image
            if file.lower().endswith(file_extensions):
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(dst_folder, file)

                # Move the file
                if os.path.exists(dst_file_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    # Ensure unique file names in the destination folder
                    while os.path.exists(dst_file_path):
                        dst_file_path = os.path.join(dst_folder, f"{base}_{counter}{ext}")
                        counter += 1

                shutil.move(src_file_path, dst_file_path)
                print(f"Moved: {src_file_path} -> {dst_file_path}")

    print("All images have been moved.")


# Define source and destination folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
src_folder = os.path.join(BASE_DIR, 'Dataset', 'train', 'Images')  # Source folder with subfolders
dst_folder = os.path.join(BASE_DIR, 'Dataset', 'train', 'images')  # Destination folder

# Call the function to move images
move_images(src_folder, dst_folder)
