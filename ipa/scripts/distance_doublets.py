import numpy as np
import pandas as pd
import argparse


def compute_dist_group(df, pixel_sizeinit):
    def compute_dist_for_group(group,pixel_sizeinit):
        d1 = np.sqrt(np.sum((np.diff(group[group.channel == 0][['x', 'y', 'z', 'snr_original']].sort_values(by='snr_original', ascending=False).iloc[0:2].values[:, 0:3], axis=0) *  pixel_sizeinit) ** 2))
        d2 = np.sqrt(np.sum((np.diff(group[group.channel == 1][['x', 'y', 'z', 'snr_original']].sort_values(by='snr_original', ascending=False).iloc[0:2].values[:, 0:3], axis=0) * pixel_sizeinit) ** 2))

        if d1 == 0 or d2 == 0:
            return pd.Series([None, None, group.frame.iloc[0], group.label.iloc[0]])

        return pd.Series([d1, d2, group.frame.iloc[0], group.label.iloc[0]])

    grouped = df.groupby(['frame', 'label'])[df.columns]
    result = grouped.apply(lambda group: compute_dist_for_group(group.reset_index(drop=True), pixel_sizeinit))
    result.columns = ['d1', 'd2', 'frame', 'label']
    return result.reset_index(drop=True)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=str, help="Path to the nd2 file")
    parser.add_argument("--output_file", type=str, help="Output file",default="distance_max2.csv")
    parser.add_argument("--pixel_sizeinit", type=float, nargs='+', help="Pixel size in x,y,z", default=[0.13,0.13,0.3])

    args = parser.parse_args()

    detections = pd.read_parquet(args.detections)
    
    if len(detections)==0:
        print('No detections found')
        return pd.DataFrame().to_parquet(args.output_file, index=False)
    
    df = compute_dist_group(detections,args.pixel_sizeinit)

    df.filename = args.detections

    df.to_parquet(args.output_file, index=False)

if __name__ == '__main__':
    
    main()