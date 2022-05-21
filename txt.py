import os

path_imgs = './data/moderate/train/input/'
fs = os.listdir(path_imgs)
fs.sort(key=lambda x: int(x.split('.')[0]))
for files in fs:
    print(files)
    img_path = files

    with open("data/moderate/train/train.txt", "a") as f:
        f.write(str(img_path) + '\n')
