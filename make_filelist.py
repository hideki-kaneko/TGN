import os
import csv
import argparse
from PIL import Image
import numpy as np

def make_train_data_list(root_dir, dst_path, ignored_dst_path, minth, maxth, delimiter=" "):
    f1 = open(dst_path, 'w')
    f2 = open(ignored_dst_path, 'w')
    writer1 = csv.writer(f1, delimiter=delimiter)
    writer2 = csv.writer(f2, delimiter=delimiter)
    for cur_dir, dirs, files in os.walk(root_dir):
        print(cur_dir)
        for file in files:
            if file.endswith(".jpg"):
                x_path = os.path.join(cur_dir, file)
                x_path_short = x_path[len(root_dir):]
                y_label = cur_dir[cur_dir.rfind("/")+1:]
                data = [x_path_short, y_label]
                img = Image.open(x_path)
                h,s,v = img.convert("HSV").split()
                hue = float(np.asarray(h).mean())
                if minth <= hue and hue <= maxth:
                    writer1.writerow(data)
                else:
                    writer2.writerow(data)
    f1.close()
    f2.close()
                        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir",
                       help="specify the root directory")
    parser.add_argument("dst_path",
                       help="specify destination file path")
    parser.add_argument("ignored_dst_path")
    parser.add_argument("minth", type=int)
    parser.add_argument("maxth", type=int)
    parser.add_argument("--delimiter",
                       help="delimiter in the output csv",
                       default = " ")
    
    args = parser.parse_args()
    
    make_train_data_list(args.root_dir, args.dst_path, args.ignored_dst_path,
                        args.minth, args.maxth, args.delimiter)
    
if __name__ == "__main__":
    main()