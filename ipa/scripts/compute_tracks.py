import argparse
import pandas as pd
import numpy as np
from pathlib import Path
<<<<<<< HEAD
from analysis_utils import process_df
=======
>>>>>>> 23a9f7447a39fc8b26ef63de828bdf95784621bd
from create_tracks_no_beads import process_df_no_beads
import re

def compute_tracks(input_file,cutoff,proportion_good_track,method ,output_file,cxy,cz,raw):

    # Compute the tracks
    radial_distances,trajs_c1,trajs_c2,labels_to_save,snr_1,snr_2,N_pixel,files = process_df_no_beads(input_file,cutoff=cutoff,proportion_good_track=proportion_good_track,method=method,cxy=cxy,cz=cz,raw=raw)
    
    pattern = r'\d{8}'

    dates = re.search(pattern, files).group(0)

    distances_total=[]
    pos_c1=[]
    pos_c2=[]
    pixel=[]
    dat = []
    labels = []
    for i in range(len(radial_distances)):
        index=np.abs(radial_distances[i][:,0])>0
        
        data_to_append=np.zeros((np.sum(index),9))
        data_to_append[:,0]=snr_1[i][index,0].flatten()
        data_to_append[:,1]=snr_1[i][index,1].flatten() * 0.13
        data_to_append[:,2]=snr_1[i][index,2].flatten() * 0.3
        
        data_to_append[:,3]=snr_2[i][index,0].flatten()
        data_to_append[:,4]=snr_2[i][index,1].flatten()* 0.13
        data_to_append[:,5]=snr_2[i][index,2].flatten()* 0.3
        
        data_to_append[:,6]=radial_distances[i][index,0]
        data_to_append[:,7]=radial_distances[i][index,1]
        data_to_append[:,8]=radial_distances[i][index,2] 

        distances_total.extend(data_to_append)
        pos_c1.extend(trajs_c1[i][index])
        pos_c2.extend(trajs_c2[i][index])
        pixel.extend(N_pixel[i][index])
        dat.extend([dates]*np.sum(index))
        labels.extend(labels_to_save[i][index])
    
    try:
        df_temp = np.vstack([x for x in distances_total if len(x)>1])
        pos = np.vstack([x for x in pos_c1 if len(x)>1])
        print(len(pos[0]))
        pos2 = np.vstack([x for x in pos_c2 if len(x)>1])
        df = pd.DataFrame(df_temp,columns=['snr_1','sigma1x','sigmaz','snr_2','sigma2x','sigma2z','dx','dy','dz'])
        df['dates'] = dat
        df[['x', 'y','z','x_fitted_refined', 'y_fitted_refined','z_fitted_refined']]= pos
        df[['x', 'y','z','x_fitted_refined_c2', 'y_fitted_refined_c2','z_fitted_refined_c2']]= pos2
        df[['label','new_label','frame']]= labels
        # Save the output file
        return df.to_parquet(output_file, index=False)
    except ValueError:
        print('No tracks found')
        return pd.DataFrame().to_parquet(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute tracks from a file')
    parser.add_argument('--input_file', type=str, help='The input file')
    parser.add_argument('--cutoff', type=float, help='The cutoff')
    parser.add_argument('--proportion_good_track', type=float, help='The proportion_good_track')
    parser.add_argument('--method', type=str, help='The method')
    parser.add_argument('--output_file', type=Path, help='The output file')
    parser.add_argument('--crop_size_xy', type=int, help='The cxy')
    parser.add_argument('--crop_size_z', type=int, help='The cz')
    parser.add_argument('--raw', type=bool, help='The raw')

    args = parser.parse_args()

    if args.raw == 'True':
        args.raw = True
    else:
        args.raw = False

    compute_tracks(args.input_file, args.cutoff, args.proportion_good_track, args.method, args.output_file, args.crop_size_xy, args.crop_size_z, args.raw)

