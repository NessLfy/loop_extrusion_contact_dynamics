import numpy as np
import pandas as pd
from multiprocessing import Pool
from skimage.morphology import ball, dilation
import ipa.src.snakemake_utils as snk
import tifffile
import dask.array as da
from skimage.morphology import disk,white_tophat
from trackpy.preprocessing import lowpass
from itertools import starmap
import sys
sys.path.append("/tungstenfs/scratch/ggiorget/nessim/2_color_imaging/localization_precision_estimation/ipa/src")
from localization_utils import locate_com 
import trackpy as tp
import argparse
import tqdm
import os
from scipy.ndimage import maximum_filter
import dask.config
import cProfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent
from joblib import Parallel, delayed




# function to filter the image

def format(im):
     im = white_tophat(lowpass(im,1),footprint=np.expand_dims(disk(2),axis=0))
     return im

def max_filter(raw_im):
    footprint=ball(7)
    max_filter=maximum_filter(raw_im,footprint=footprint)
    return max_filter

# function to perform the detections

def max5_detection(raw_im: np.ndarray,frame: int,channel:int,max_filter_image:np.array,labels:np.array,crop_size_xy:int = 4,crop_size_z:int = 4) -> pd.DataFrame:
    """_summary_

    Args:
        raw_im (np.array): the raw image to segment
        frame (int): the frame to segment
        sd (float): the sd of the peak intensity (threshold of segmentation)
        n (int, optional): how much brighter than the sd of the whole image to threshold
        thresh (float, optional): threshold for the gaussian fitting filter. Filter on the standard deviation of the fit (based on the covariance of the parameters) Defaults to 0.5.

    Returns:
        pd.DataFrame: Dataframe of sub-pixel localizations of the detected spots
    """
    labs = []
    max_coords_2 = []
    n_pixels = []
    snr = []
    snr_d = []
    for l in np.unique(labels):
        if l == 0:
            continue
        else:
            im_masked_max = max_filter_image * (labels == l)

            raw_im_masked_ = raw_im * (labels == l)

            raw_im_masked = raw_im_masked_*(raw_im_masked_ == im_masked_max)

            new_im_flat = raw_im_masked.flatten()
            flat_index = np.argsort(new_im_flat)[-5:]
            intensity = new_im_flat[flat_index]
            back_d = np.mean(raw_im_masked[raw_im_masked>0])
            back = np.mean(raw_im_masked_[raw_im_masked_>0])

            snr_d.extend([i/back_d for i in intensity])
            snr.extend([i/back for i in intensity])

            max_coords_2.extend([np.unravel_index(f, raw_im_masked.shape) for f in flat_index])

            n_pixels.extend([np.sum(labels == l)]*5)
            labs.extend([l]*5)

    try:
        z,y,x = np.array(max_coords_2).T
    except ValueError as e:
        print(f'The following error has occured when trying to extract segmented pixels {e}')


    # remove duplicates
    pos=np.vstack((z,y,x)).T
    margin=np.array((crop_size_z//2,crop_size_xy//2,crop_size_xy//2))
    shape=raw_im.shape
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]
    labs = np.array(labs)[~near_edge]
    n_pixels = np.array(n_pixels)[~near_edge]
    snr = np.array(snr)[~near_edge]
    snr_d = np.array(snr_d)[~near_edge]
    z,y,x = pos.T

    
    intensity = [raw_im[z,y,x] for (z,y,x) in zip(z,y,x)]


    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print("No spots detected")
        return pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','channel','intensity','pixel_sum','label','snr_tophat','snr_dilated'])

    try:
        k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z) for i in range(len(pos))]
    except IndexError as e:
        print(f'The following error has occured when trying to create the arguments for the fitting {e}')
        return pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','channel','intensity','pixel_sum','label','snr_tophat','snr_dilated'])
    
    try:
        x_s,y_s,z_s= zip(*(starmap(locate_com,k)))
    except IndexError as e:
        print(f'An error occurred when trying to map the locate_com function: {e}')
        return pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','channel','intensity','pixel_sum','label','snr_tophat','snr_dilated'])
        
    df_loc = pd.DataFrame([x,y,z,x_s,y_s,z_s]).T
    df_loc.rename(columns={0:'x',1:'y',2:'z',3:'x_fitted',4:'y_fitted',5:'z_fitted'},inplace=True)

    try:
        detections_refined= tp.refine.refine_com(raw_image = raw_im, image= raw_im, radius= [crop_size_z//2,crop_size_xy//2,crop_size_xy//2], coords = df_loc, max_iterations=1,
        engine='python', shift_thresh=0.6, characterize=True,
        pos_columns=['z_fitted','y_fitted','x_fitted'])
    except ValueError:
        detections_refined = df_loc

    df_loc[['z_fitted_refined','x_fitted_refined','y_fitted_refined']] = detections_refined[['z_fitted','x_fitted','y_fitted']]

    df_loc['frame'] = frame
    df_loc['channel'] = channel
    df_loc['intensity'] = intensity
    df_loc['pixel_sum'] = n_pixels
    df_loc['label'] = labs
    df_loc['snr_tophat'] = snr
    df_loc['snr_dilated'] = snr_d

    return df_loc


# function to load and save the data

def processing(im,output_file_path,frame,channel,labels,crop_xy,crop_z):
    if os.path.exists(output_file_path):
        return None
    else:
        im_filtered = format(im)
        max_filter_image = max_filter(im_filtered)

        spot_df = max5_detection(raw_im= im_filtered,frame= frame,channel=channel,max_filter_image=max_filter_image,labels=labels,crop_size_xy= crop_xy,crop_size_z= crop_z)

        spot_df.to_csv(output_file_path,index=False)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run detections')

    # Add the arguments
    parser.add_argument('--input_image', type=str, help='The input image file')
    parser.add_argument('--output_file', type=str, help='The output file')
    parser.add_argument('--log_filename', type=str, help='The log filename')
    parser.add_argument('--log_path', type=str, help='The log path')
    parser.add_argument('--threads', type=int, help='The number of threads')
    parser.add_argument('--labels', type=str, help='label_image')
    parser.add_argument('--n_frames', type=int, help='frame')
    parser.add_argument('--crop_size_xy', type=int, help='crop_size_xy')
    parser.add_argument('--crop_size_z', type=int, help='crop_size_z')

    args = parser.parse_args()

    input_path = args.input_image
    dask.config.set(scheduler='threads', num_workers=1)

    threads = args.threads
    
    im = da.from_zarr(input_path, component='0/')

    output_path = (args.output_file).strip('.csv') 
    
    labels = np.load(args.labels)
    n_frames = np.shape(im)[0]

    tasks = [
    (im[frame,:,channel,...], f"{output_path}_{frame}_{channel}.csv", frame, channel, labels[frame, ...], args.crop_size_xy, args.crop_size_z)
    for frame in range(n_frames) for channel in range(2)]

    Parallel(n_jobs=threads, prefer="processes", verbose=6)(
    delayed(processing)(*task) for task in tqdm.tqdm(tasks, desc="Processing images")
    )
    
    # futures = []
    # with ProcessPoolExecutor(max_workers=threads) as executor:
    #     for task in tasks:
    #         # Unpack the arguments from the task tuple and submit them to the executor
    #         future = executor.submit(processing, *task)
    #         futures.append(future)

    #     _ = [future.result() for future in tqdm.tqdm(as_completed(futures), total=len(futures))]

    df_final = []
    for files in tqdm.tqdm(os.listdir('/'.join(output_path.split('/')[:-1]))):
        if files.endswith('.csv'):
            if output_path.split('/')[-1] in files:
                try:
                    df_temp = pd.read_csv('/'.join(output_path.split('/')[:-1])+'/'+files)
                    df_final.append(df_temp)
                    os.remove('/'.join(output_path.split('/')[:-1])+'/'+files)
                except UnicodeDecodeError:
                    print(f'Error reading {files}')
                    continue

    df_final = pd.concat(df_final)

    df_final.to_parquet(output_path+'.csv',index=False)

if __name__ == '__main__':
    cProfile.run('main()', '/tungstenfs/scratch/ggiorget/nessim/2_color_imaging/localization_precision_estimation/runs/20240725_test_speed/profile_output_ThreadPoolExecutor.dat')
