import os
import glob


if __name__ == '__main__':
    color2code = {
        '緑': 'g',
        '青': 'b',
        '紫': 'p',
        '黒': 'k',
        '黄色': 'y'
    }

    base_path = "../../data/"
    for key, value in color2code.items():
        files = glob.glob(base_path + key + "/*.jpg")
        for f in files:
            os.rename(f, os.path.join(base_path, value + os.path.basename(f)))
