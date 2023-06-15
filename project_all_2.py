import os

def select_files(img_dir, subfolders):
    selected_files = []
    for subfolder in subfolders:
        file_pattern = os.path.join(img_dir, subfolder)
        file = os.path.join(file_pattern, 'projected_w.npz')
        selected_files.append(file)
    return selected_files

def list_subfolders(folder_path):
    subfolder_names = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolder_names.append(item)
    return subfolder_names

if __name__ == '__main__':
    data_dir = 'frackles'
    img_dir = os.path.join(data_dir,'anti-latent')
    out_dir = os.path.join(data_dir,'anti-latent_render_projected_w')

    os.makedirs(out_dir,exist_ok=True)
    
    subfolders = list_subfolders(img_dir)
    files_path = select_files(img_dir, subfolders)
    print({files_path[0]})
 
    # Print the list of files
    for file in files_path:
        print('START: ',out_dir)
        os.system(f'python stylegan2/generate.py --outdir={out_dir} --projected-w={file} --network=ffhq.pkl')
        print('DONE: ',out_dir)
