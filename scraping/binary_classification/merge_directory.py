import os
import glob


def change_name():
    base_dir = "/home/arai/zipfiles/classify/"
    data_dir = "/home/arai/zipfiles/weben/"
    top_dir = os.listdir(data_dir)
    top_dir.remove("cleanup")
    top_dir.remove("二値分類CNN重み")
    for dir in top_dir:
        layer1 = data_dir + f"{dir}/"
        layer1_files = os.listdir(layer1)
        save_jpg(base_dir, layer1, layer1_files, dir)
        next_dirs = [f for f in layer1_files if ".jpg" not in f]
        if len(next_dirs) != 0:
            for nd in next_dirs:
                layer2 = layer1 + f"{nd}/"
                layer2_files = os.listdir(layer2)
                dir2 = f"{dir}_{nd}"
                save_jpg(base_dir, layer2, layer2_files, dir2)
                even_deeper = [f for f in layer2_files if ".jpg" not in f]
                if len(even_deeper) != 0:
                    for td in even_deeper:
                        layer3 = layer2 + f"{td}/"
                        layer3_files = os.listdir(layer3)
                        dir3 = f"{dir2}_{td}"
                        save_jpg(base_dir, layer3, layer3_files, dir3)



def save_jpg(base_path, layer_path, layer_files, dir):
    jpgs = [(layer_path + file) for file in layer_files if ".jpg" in file]
    save = [(base_path + f"{dir}_{file}") for file in layer_files if ".jpg" in file]
    if len(jpgs) != 0:
        for j, s in zip(jpgs, save):
            os.rename(j, s)
