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

    all_dist = [[0, 199.54842125930608, 164.7777285075194, 218.65710729190812, 223.72032871651703, 196.94137469220655, 242.4490490932918, 218.00850443806274, 84.6062018465674, 215.58522675914372, 171.2986408409686, 173.9757736960937], [199.54842125930608, 0, 148.6823684422471, 191.6860007681375, 246.4260593925083, 198.5658986195124, 220.0158363410316, 221.62224186784272, 211.38301755233962, 193.30837619468056, 146.44388060301767, 172.1299558643671], [164.7777285075194, 148.6823684422471, 0, 161.50128452802767, 195.738884898267, 193.26423087928615, 189.47165204356776, 186.3446428551176, 174.86914621121235, 162.6063483420174, 125.49208151073775, 126.73489762287151], [218.65710729190812, 191.6860007681375, 161.50128452802767, 0, 253.58026309779765, 232.31251378583315, 223.92316247810965, 240.40427982326662, 228.96515895171555, 204.088341288946, 162.12222662776063, 176.76727542978566], [223.72032871651703, 246.4260593925083, 195.738884898267, 253.58026309779765, 0, 263.0060707627875, 263.8019032358997, 263.5435383801827, 234.81556682376177, 247.8106519500269, 218.23475975129992, 207.22633494383876], [196.94137469220655, 198.5658986195124, 193.26423087928615, 232.31251378583315, 263.0060707627875, 0, 256.49119252036235, 232.36700536271204, 209.46108295377752, 230.67922248998715, 202.1048953582001, 210.06279660289766], [242.4490490932918, 220.0158363410316, 189.47165204356776, 223.92316247810965, 263.8019032358997, 256.49119252036235, 0, 264.19726611375796, 252.38701520464215, 252.02312548758061, 185.59970423798137, 202.18732933850862], [218.00850443806274, 221.62224186784272, 186.3446428551176, 240.40427982326662, 263.5435383801827, 232.36700536271204, 264.19726611375796, 0, 222.13293024099957, 227.35233321786396, 209.65474361278987, 215.17777779854518], [84.6062018465674, 211.38301755233962, 174.86914621121235, 228.96515895171555, 234.81556682376177, 209.46108295377752, 252.38701520464215, 222.13293024099957, 0, 222.90126115721512, 187.44337898078615, 186.7553243739015], [215.58522675914372, 193.30837619468056, 162.6063483420174, 204.088341288946, 247.8106519500269, 230.67922248998715, 252.02312548758061, 227.35233321786396, 222.90126115721512, 0, 185.4333753222723, 189.2843081623223], [171.2986408409686, 146.44388060301767, 125.49208151073775, 162.12222662776063, 218.23475975129992, 202.1048953582001, 185.59970423798137, 209.65474361278987, 187.44337898078615, 185.4333753222723, 0, 115.33892599618306], [173.9757736960937, 172.1299558643671, 126.73489762287151, 176.76727542978566, 207.22633494383876, 210.06279660289766, 202.18732933850862, 215.17777779854518, 186.7553243739015, 189.2843081623223, 115.33892599618306, 0]]

    args = create_parser()

    cwd = os.getcwd()

    print(args.unblocked)

    in_file = "inclus_{blocked}_{c}.out".format(blocked="unblock" if args.unblocked else "blocked", c=args.cluster)
    out_file = "outclus_{blocked}_{c}.out".format(blocked="unblock" if args.unblocked else "blocked", c=args.cluster)
    add_file = "addclus_{blocked}_{c}.out".format(blocked="unblock" if args.unblocked else "blocked", c=args.cluster)

    in_c = np.loadtxt(os.path.join(cwd, in_file))
    out_c = np.loadtxt(os.path.join(cwd, out_file), dtype=str)
    add_c = np.loadtxt(os.path.join(cwd, add_file))

    dist = all_dist[int(args.cluster)]
    
    out_c_processed = []
    for oc in out_c:
        out_c_processed.append(float(oc[0]))

    #print(out_c_processed)
    print(np.mean(in_c))

    fig, ax = plt.subplots()

    in_c_counts, in_c_bins = np.histogram(in_c, 50, density=True)
    out_c_counts, out_c_bins = np.histogram(out_c_processed, 100, density=True)
    
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
