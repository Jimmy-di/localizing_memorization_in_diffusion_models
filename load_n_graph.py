import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Graph Results')

    parser.add_argument(
        '-f',
        '--filename',
        default='test.png', 
        type=str,
        dest="fname",
        help='Graph file save path\').'
    )

    parser.add_argument(
        '-c',
        '--cluster',
        default=0, 
        type=int,
        dest="cluster",
        help='Cluster Number\').'
    )

    parser.add_argument(
        '-u',
        '--unblocked',
        action='store_true', 
        default=False, 
        help='Blocked neurons or Unblocked Neurons'
    )

    args = parser.parse_args()
    return args


def main():
    args = create_parser()

    cwd = os.getcwd()

    print(args.unblocked)

    in_file = "inclus_{blocked}_{c}.out".format(blocked="unblock" if args.unblocked else "block", c=args.cluster)
    out_file = "outclus_{blocked}_{c}.out".format(blocked="unblock" if args.unblocked else "block", c=args.cluster)
    add_file = "addclus_{blocked}_{c}.out".format(blocked="unblock" if args.unblocked else "block", c=args.cluster)

    in_c = np.loadtxt(os.path.join(cwd, in_file))
    out_c = np.loadtxt(os.path.join(cwd, out_file))
    add_c = np.loadtxt(os.path.join(cwd, add_file))

    print(np.mean(in_c))

    fig, ax = plt.subplots()

    in_c_counts, in_c_bins = np.histogram(in_c, 50, density=True)
    out_c_counts, out_c_bins = np.histogram(out_c, 100, density=True)
    
    if len(add_c) > 0:
        add_c_counts, add_c_bins = np.histogram(add_c, 100, density=True)
    else:
        add_c_counts = []

    ax.plot(in_c_bins[:-1], in_c_counts, label="In-Cluster")
    ax.plot(out_c_bins[:-1], out_c_counts, label="Out-Cluster")

    if len(add_c_counts) > 0:
        ax.plot(add_c_bins[:-1], add_c_counts, label="Additional Prompts")


    ax.set_ylabel('density')
    ax.set_xlabel('CLIP Similarity Score')
    ax.legend()
    ax.set_title('CLIP Similarity Score for Cluster {}'.format(args.cluster))
    ax.yaxis.grid(True)

    fig.savefig(args.fname)


if __name__ == "__main__":
    main()
