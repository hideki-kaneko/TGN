# make_dataset.py
# Usage: python3 make_dataset.py --labels_path="./path_to_label.csv" --src_dir="./path_to_image_dir" --dst_dir="path_to_output_dir"

import argparse
import csv
import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
from tqdm import tqdm


def shift(x,delta):
    return min(max(x+delta,0),255)

def change_hsv(img):
    vshift = np.vectorize(shift)
    CHANGE_HUE = True
    CHANGE_SAT = True
    CHANGE_BRT = True

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    delta = random.gauss(0,20)
    if CHANGE_HUE:
        img[:,:,0] = vshift(img[:,:,0], delta)
    delta = random.gauss(0,20)
    if CHANGE_SAT:
        img[:,:,1] = vshift(img[:,:,1], delta)
    delta = random.gauss(0,10)
    if CHANGE_BRT:
        img[:,:,2] = vshift(img[:,:,2], delta)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def crop_image(img, x, y, size):
    crop = img.crop((x, y, x+size, y+size))
    return crop

def slice_image(img_path, dst_dir, size, stride=-1, angle_step = 30):
    # 入力された画像を正方形の小領域に分割して保存します。
    # 入力画像に対して、(size, size)の正方形の領域をstrideずつxy方向にずらして切り出す操作を繰り返します。
    # 切り出しが終わるごとにangle_step分だけ入力画像を回転させ、同様の操作を繰り返します。
    # 正方形の領域のうち、入力画像を二値化して黒くなっている部分の占める割合が100%に満たないものは保存されません。
    #
    # img_path[str]     : 入力画像(jpg)の絶対パス
    # dst_dir[str]      : 分割した画像を保存するディレクトリのパス
    # size[int]         : 分割する正方形の1辺の長さ(px)
    # stride[int]       : 正方形の領域をスライドさせる幅（-1を指定した場合、正方形の1辺の長さと等しくなります）
    # angle_step[float] : 1回の切り出しで画像を回転させる角度

    if stride==-1:
        stride = size
    
    # 画像の読み込み
    img_org = Image.open(img_path)
    img_org = img_org.crop((0,0,img_org.size[0], img_org.size[1]*0.5))
    width, height = img_org.size
    x = 0
    y = 0
    angle = 0
    name = os.path.basename(img_path)[:-4]
    
    # 元画像の回転角を少しずつ変えながらループ
    while angle < 360:
        img = img_org.convert('RGBA').rotate(angle)     # 画像を回転
        white = Image.new('RGBA', img.size, (255,)*4)   # 回転させた画像を貼り付ける空白の領域を確保
        img = Image.composite(img, white, img)          # 回転させた画像を空白の領域に貼り付け
        img = img.convert('RGB')                        # 透過度を表現するアルファチャンネルは不要なので捨てる
        bi = convert_to_binary(img)                     # 画像を二値化
        img = np.asarray(img)                           # 各ピクセルの値を見たいのでnumpyの行列に変換
        # 切り出す座標(x,y)を少しずつ変えながらループ
        while y <= height-size:
            while x <= width-size:
                crop = img[y:y+size, x:x+size]              # 画像の座標(x,y)を原点とした1辺sizeの正方領域を切り出し
                if bi[y:y+size, x:x+size].mean() < 1.0:     # 正方領域のうち黒い部分が占める割合が100%に満たない場合はスキップ
                    pass
                else:
                    crop_img = Image.fromarray(crop)
                    # crop_img = change_hsv(crop)
                    cv2.imwrite(dst_dir + "/" + name + "_x" + str(x) + "y" + str(y) + "r" + str(angle) + ".jpg", crop_img)  # 適当な名前をつけて切り出した部分を保存
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
    random.seed(123)
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
