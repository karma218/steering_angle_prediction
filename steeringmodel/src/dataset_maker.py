import os
import shutil
import pandas as pd

curr_dir = os.getcwd()

# Define the path to the data folder
dataset_folder = os.path.join(curr_dir,"../Dataset")


# Define the paths to the image and CSV files
center_images_folder = os.path.join(dataset_folder, "center")
steering_data_path = os.path.join(dataset_folder, "steering.csv")


final_images_folder_path = os.path.join(dataset_folder, 'final_images_center')
if not os.path.exists(final_images_folder_path):
        os.mkdir(final_images_folder_path)

# Read the CSV files
steering_data = pd.read_csv(steering_data_path)


steering_data['timestamp']  = steering_data['timestamp'].astype(str).str[:-7]

print(steering_data)

steering_ts_ls = steering_data['timestamp'].tolist()
steering_angle_ls = steering_data['angle'].tolist()

# Iterate over each image file in the images folder
def make_dataset_from_image_path(path):
    timestamp_list = []
    angle_list = []
    # throttle_list = []
    # brake_list = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_name_as_ts = os.path.splitext(filename)[0][:-7]
            if image_name_as_ts in steering_ts_ls:

                # Copy the file to the finalImages folder
                source_file = os.path.join(path, filename)
                destination_file = os.path.join(final_images_folder_path, image_name_as_ts + ".png")
                shutil.copy(source_file, destination_file)

                steering_row = steering_data[steering_data['timestamp'] == image_name_as_ts]
            
                def find_data(row, cname):
                    if not row.empty:
                        data = row[cname].iloc[0]
                        return data

                angle = find_data(steering_row, 'angle')
                timestamp_list.append(image_name_as_ts)
                angle_list.append(angle)

            
    return pd.DataFrame({'timestamp': timestamp_list, 'angle': angle_list}) 

center = make_dataset_from_image_path(center_images_folder)
final_data = pd.concat([center], ignore_index=True)


# Write the final DataFrame to a CSV file
final_data.to_csv(os.path.join(dataset_folder,'final_data_center.csv'), index=False)

# print(final_data)
