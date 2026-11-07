import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(BASE_DIR, 'Dataset', 'train', 'Images')  # Directory with subfolders for each class
label_folder = os.path.join(BASE_DIR, 'Dataset', 'train', 'labels')  # Directory where annotation files will be saved

# Class names (update according to your dataset)
class_names = [
    'Give way', 'No entry', 'One-way traffic', 'One-way traffic', 'No vehicles in both directions',
    'No entry for cycles', 'No entry for goods vehicles', 'No entry for pedestrians',
    'No entry for bullock carts', 'No entry for hand carts', 'No entry for motor vehicles',
    'Height limit', 'Weight limit', 'Axle weight limit', 'Length limit', 'No left turn',
    'No right turn', 'No overtaking', 'Maximum speed limit (90 km/h)', 'Maximum speed limit (110 km/h)',
    'Horn prohibited', 'No parking', 'No stopping', 'Turn left', 'Turn right', 'Steep descent',
    'Steep ascent', 'Narrow road', 'Narrow bridge', 'Unprotected quay', 'Road hump', 'Dip',
    'Loose gravel', 'Falling rocks', 'Cattle', 'Crossroads', 'Side road junction', 'Side road junction',
    'Oblique side road junction', 'Oblique side road junction', 'T-junction', 'Y-junction',
    'Staggered side road junction', 'Staggered side road junction', 'Roundabout',
    'Guarded level crossing ahead', 'Unguarded level crossing ahead', 'Level crossing countdown marker',
    'Level crossing countdown marker', 'Level crossing countdown marker', 'Level crossing countdown marker',
    'Parking', 'Bus stop', 'First aid post', 'Telephone', 'Filling station', 'Hotel', 'Restaurant',
    'Refreshments'
]

# Create label directory if it doesn't exist
os.makedirs(label_folder, exist_ok=True)

# Iterate over each class folder
for class_folder in os.listdir(image_folder):
    class_folder_path = os.path.join(image_folder, class_folder)
    if os.path.isdir(class_folder_path):
        class_id = int(class_folder)  # Assuming folder names are numeric class IDs

        # Iterate over each image in the class folder
        for image_file in os.listdir(class_folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Derive the label file name
                base_name = os.path.splitext(image_file)[0]
                label_file = os.path.join(label_folder, f'{base_name}.txt')

                # Write annotation file
                with open(label_file, 'w') as file:
                    # For fully cropped images, bounding box covers entire image
                    file.write(f'{class_id} 0.5 0.5 1.0 1.0\n')

print('Annotation files have been created.')
