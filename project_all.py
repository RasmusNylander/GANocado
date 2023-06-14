import os

data_dir = 'frackles'
img_dir = os.path.join(data_dir,'frackles')
out_dir = os.path.join(data_dir,'frackles_latent')

os.makedirs(out_dir,exist_ok=True)

def list_files_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid folder path.")
        return

    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_list.append(file_path)

    return file_list


# Call the function and get the list of files
files = list_files_in_folder(img_dir)

# Print the list of files
for file in files:
    os.system(f'python stylegan2/projector.py --network=ffhq.pkl --outdir={out_dir} --target={file} --save-video False')

