import os
import csv
import argparse
import pandas as pd

def make_train_data_list(csv_path, root_dir, dst_path, delimiter=" "):
    with open(dst_path, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        df = pd.read_csv(csv_path)
        for cur_dir, dirs, files in os.walk(root_dir):
            print(cur_dir)
            for file in files:
                if file.endswith(".jpg"):
                    x_name = file
                    x_path = os.path.join(cur_dir, x_name)
                    x_path_short = x_path[len(root_dir):]
                    if x_path_short[0] == "/":
                         x_path_short = x_path_short[1:]
                    y_path = os.path.join(cur_dir, file)
                    y_path_short = y_path[len(root_dir):]
                    dirID = int(cur_dir[cur_dir.rfind("/")+1:])
                    classID = int(df[df["ID"]==dirID]["CLASS"])
                    if classID != 12: #UNKNOWN
                        row = [x_path_short, classID]
                        writer.writerow(row) 
                        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("root_dir",
                       help="specify the root directory")
    parser.add_argument("dst_path",
                       help="specify destination file path")
    parser.add_argument("--delimiter",
                       help="delimiter in the output csv",
                       default = " ")
    args = parser.parse_args()
    
    make_train_data_list(args.csv, args.root_dir, args.dst_path, args.delimiter)
    
if __name__ == "__main__":
    main()

