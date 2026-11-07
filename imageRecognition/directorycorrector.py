import os
import shutil

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(BASE_DIR, 'Dataset', 'train', 'Images')  # Main image folder
label_folder = os.path.join(BASE_DIR, 'Dataset', 'train', 'labels')  # Main label folder
output_label_folder = os.path.join(BASE_DIR, 'Dataset', 'train', 'newlabels')  # Where you want to move labels

# Create the output label directory if it doesn't exist
os.makedirs(output_label_folder, exist_ok=True)

# Iterate over the class folders
for class_folder in os.listdir(image_folder):
    class_folder_path = os.path.join(image_folder, class_folder)
    if os.path.isdir(class_folder_path):
        # Create corresponding directory in output label folder
        output_class_folder = os.path.join(output_label_folder, class_folder)
        os.makedirs(output_class_folder, exist_ok=True)

        # Move label files
        for label_file in os.listdir(label_folder):
            if label_file.endswith('.txt'):
                # Derive the image base name (remove leading zeros and extension)
                base_name = label_file.lstrip('0').split('.')[0]

                # Construct paths
                src_label_path = os.path.join(label_folder, label_file)
                dst_label_path = os.path.join(output_class_folder, f'{base_name}.txt')

                # Move label file
                shutil.move(src_label_path, dst_label_path)

print('Labels have been moved to the appropriate directories.')
