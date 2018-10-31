import os
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
import pandas as pd
from tqdm import tqdm
import csv
import argparse

def crop_image(img, x, y, size):
    crop = img.crop((x, y, x+size, y+size))
    return crop

def slice_image(img_path, dst_dir, size, stride=-1, angle_step = 30):
    if stride==-1:
        stride = size
    img_org = Image.open(img_path)
    img_org = img_org.crop((0,0,img_org.size[0], img_org.size[1]*0.5))
    width, height = img_org.size
    x = 0
    y = 0
    angle = 0
    name = os.path.basename(img_path)[:-4]
    
    while angle < 360:
        img = img_org.convert('RGBA').rotate(angle)
        white = Image.new('RGBA', img.size, (255,)*4)
        img = Image.composite(img, white, img)
        img = img.convert('RGB')
        bi = convert_to_binary(img)
        img = np.asarray(img)
        while y <= height-size:
            while x <= width-size:
                crop = img[y:y+size, x:x+size]
                #h,s,v = Image.fromarray(crop).convert("HSV").split()
                #hue = float(np.asarray(h).mean())
#                 if bi[y:y+size, x:x+size].mean() < 1.0 or hue<55 or hue>65:
                if bi[y:y+size, x:x+size].mean() < 1.0:
                    pass
                else:
                    crop_img = Image.fromarray(crop)
                    crop_img.save(dst_dir + "/" + name + "_x" + str(x) + "y" + str(y) + "r" + str(angle) + ".jpg")
                x+=stride
            x=0
            y+=stride
        y=0
        angle += angle_step
    

def convert_to_binary(img, thres=-1):
    img = np.asarray(img.convert("L"))
    if thres==-1:
        thres = threshold_otsu(img)
    img = (img < thres).astype(np.int32)
    return img

def make_dataset(base_dir,out_dir, labels):
    dirs = os.listdir(base_dir)
    for d in tqdm(dirs):
        if d in labels:
            for f in os.listdir(base_dir + "/" + d):
                out_dir_path = out_dir + "/" + d
                if not os.path.exists(out_dir_path):
                    #print(out_dir_path)
                    os.makedirs(out_dir_path)
                if f[-3:] == "jpg":
                    src_img = base_dir + "/" + d + "/" + f                    
                    slice_image(src_img, out_dir_path, 256)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path")
    parser.add_argument("src_dir")
    parser.add_argument("dst_dir")
    args = parser.parse_args()
    
    df = pd.read_csv(args.labels_path)
    names = set(df["NAME"])
    name_to_class = {j:i for i, j in enumerate(names)}
    id_to_class = {i[0]:name_to_class[i[1]] for i in np.asarray(df)}
    make_dataset(args.src_dir, args.dst_dir, str(id_to_class.keys()))
                    
if __name__ == "__main__":
    main()
