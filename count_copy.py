import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse


def main():

    in_concealment = []
    out_concealment = []
    in_total = 0
    out_total = 0

    in_copy = "inclus_blocked_"
    file_end = "_0.428.out"
    out_copy = "outclus_blocked_"
    threshold = 0.85

    cwd = os.getcwd()

    for i in range(0, 21):
        
        in_file = in_copy + str(i) + file_end
        out_file = out_copy + str(i) + file_end

        in_c = np.loadtxt(os.path.join(cwd, in_file), dtype=str)
        out_c = np.loadtxt(os.path.join(cwd, out_file), dtype=str)

        in_total+=len(in_c)
        out_total+=len(out_c)

        #print(in_c)

        for c in out_c:
            c[1] = str(i)+"_" + c[1]
            #print(c[0])
            if float(c[0]) >= threshold:
                out_concealment.append(c) 
        
        for c in in_c:
            
            img_name = c[1]
            train_name = c[2]

            if int(img_name.split("_")[1]) == int(train_name.split("_")[1].strip(".jpg")):
                continue
            elif float(c[0]) >= threshold:
                c[1] = str(i)+"_" + c[1]
                in_concealment.append(c) 

    print(len(in_concealment))
    print(len(out_concealment))
    print(in_total)
    print(out_total)

        
        

    


if __name__ == "__main__":
    main()
