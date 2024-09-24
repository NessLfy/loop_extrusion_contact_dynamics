import nd2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from multiprocessing import Pool
import os
import pandas as pd
import sys

sys.path.append('/tungstenfs/scratch/ggiorget/nessim/2_color_imaging/localization_precision_estimation/ipa/src/')
from localization_utils import find_start_end,gauss_single_spot_2d_1d

from itertools import starmap
import trackpy as tp
from concurrent.futures import ThreadPoolExecutor,as_completed,ProcessPoolExecutor


from skimage.morphology import ball
from skimage.morphology import disk,white_tophat
from scipy.ndimage import maximum_filter
from trackpy.preprocessing import lowpass
import concurrent.futures

def format_im(im):
     im = white_tophat(lowpass(im,1),footprint=np.expand_dims(disk(2),axis=0))
     return im

def max_filter(raw_im):
    footprint=ball(7)
    max_filter=maximum_filter(raw_im,footprint=footprint)
    return max_filter

def locate_z(
    image: np.ndarray,
    y_coord: int,
    x_coord: int,
    z_coord: int,
    crop_size_z: int,
    thresh=0.6):    
    
    start_dim3, end_dim3 = find_start_end(z_coord, image.shape[0], crop_size_z)
    
    z_c1=np.arange(z_coord-crop_size_z//2,z_coord+crop_size_z//2+1)
    
    crop = image[start_dim3:end_dim3,y_coord,x_coord]
    
    z=np.sum(z_c1*crop/np.sum(crop))
 
    if((z-z_coord)<-thresh and z_coord-crop_size_z//2-1>=0):
        z_coord-=1
        
        start_dim3, end_dim3 = find_start_end(z_coord, image.shape[0], crop_size_z)
 
        z_c1=np.arange(z_coord-crop_size_z//2,z_coord+crop_size_z//2+1)
    
        crop = image[start_dim3:end_dim3,y_coord,x_coord]
    
        z=np.sum(z_c1*crop/np.sum(crop))
    
    elif((z-z_coord)>thresh and z_coord+crop_size_z//2+1<=crop_size_z-1):
        z_coord+=1
        
        start_dim3, end_dim3 = find_start_end(z_coord, image.shape[0], crop_size_z)
 
        z_c1=np.arange(z_coord-crop_size_z//2,z_coord+crop_size_z//2+1)
    
        crop = image[start_dim3:end_dim3,y_coord,x_coord]
 
        z=np.sum(z_c1*crop/np.sum(crop))
    
    return z

def process_frame(im,df,frame,channel,method='gaussian'):

    # im,df,frame,channel = inputs

    raw_im = nd2.imread(im,dask=True)

    if len(raw_im.shape) != 5:
        raw_im = raw_im.reshape((raw_im.shape[0]//8,8,2,raw_im.shape[-2],raw_im.shape[-1]))
    
    raw_im = raw_im[frame,:,channel,...]

    raw_im = format_im(raw_im)

    crop_size_xy = 9

    crop_size_z = 7

    snr = df[(df.frame == frame)&(df.channel == channel)][['snr_tophat','snr_dilated']].values

    lab = df[(df.frame == frame)&(df.channel == channel)]['label'].values

    loc_old = df[(df.frame == frame)&(df.channel == channel)][['z_fitted_refined','y_fitted_refined','x_fitted_refined']].values

    pos=(df[(df.frame == frame)&(df.channel == channel)][['z','y','x']].values)

    pixel_sum_old = df[(df.frame == frame)&(df.channel == channel)][['pixel_sum']].values

    margin=np.array((crop_size_z//2,crop_size_xy//2,crop_size_xy//2))
    shape=raw_im.shape
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]
    snr = snr[~near_edge]
    loc_old = loc_old[~near_edge]
    lab = lab[~near_edge]
    pixel_sum_old = pixel_sum_old[~near_edge]

    s_t,s_o = snr.T

    z,y,x = pos.T

    z_o,y_o,x_o = loc_old.T

    df_loc = pd.DataFrame([x,y,z,lab,s_o,s_t,x_o,y_o,z_o,pixel_sum_old]).T
    df_loc.columns = ['x','y','z','label','snr_dilated','snr_tophat','x_fitted_refined_old','y_fitted_refined_old','z_fitted_refined_old','pixel_sum']

    if method == 'com':
        fitted_2d = []
        for i,row in df_loc.iterrows():
            df_temp = df_loc[(df_loc.x == row.x)&(df_loc.y == row.y)&(df_loc.z == row.z)]

            detections_refined= tp.refine.refine_com(raw_image = raw_im[int(row.z),...], image= raw_im[int(row.z),...], radius= [crop_size_xy//2,crop_size_xy//2], coords = df_temp, max_iterations=1,
            engine='python', shift_thresh=0.6, characterize=False,
            pos_columns=['y','x'])
            fitted_2d.append(detections_refined[['x','y']])

        detections_refined = pd.concat(fitted_2d)

        df_loc['x_fitted_refined'] = detections_refined['x'].values
        df_loc['y_fitted_refined'] = detections_refined['y'].values


        pos=df_loc[["z","y","x"]].values.astype(int)
        k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_z) for i in range(len(pos))]
        z_s=[]
        for i in k:
            z_s.append(locate_z(*i))

        df_loc["z_fitted_refined"] = z_s
        
    elif method == 'gaussian':

        k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z) for i in range(len(pos))]
        x_s,y_s,z_s,sx,sy,sz= zip(*(starmap(gauss_single_spot_2d_1d,k)))
        df_loc = pd.DataFrame([x,y,z,x_s,y_s,z_s,sx,sy,sz]).T
        df_loc.columns=['x','y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined',"sigma_x","sigma_y","sigma_z"]
        df_loc['label'] = lab
        df_loc['snr_dilated'] = s_o
        df_loc['snr_tophat'] = s_t

    df_loc['frame'] = frame

    df_loc['channel'] = channel

    return df_loc

def process_file(inputs):
    im_file,det_file = inputs
    im = nd2.imread(im_file,dask=True)

    if os.path.isfile(det_file) == False:
        print(f"File {det_file} does not exist")
        return None

    else:
        df = pd.read_parquet(det_file)

    if os.path.isfile(det_file.replace('detections_new_crop','detections_gaussian_2D1D')):
        print(f"Already processed {im_file}")
        return None 

    else:
        frame_channel_combinations = [(im_file, df, frame, channel) for frame in range(np.shape(im)[0]) for channel in range(2)]

        # Initialize a progress bar with the total number of tasks
        progress_bar = tqdm(total=len(frame_channel_combinations), desc="Processing frames")
        results = []
        with ProcessPoolExecutor(max_workers=50) as executor:
            # Submit all tasks and keep track of the futures
            futures = [executor.submit(process_frame, *args) for args in frame_channel_combinations]
            
            # As each future is completed, update the progress bar
            for future in as_completed(futures):
                results.append(future.result())  # You can collect or process results here
                progress_bar.update(1)  # Update the progress bar by one for each completed task

        progress_bar.close() 

        df = pd.concat(results)

        df.to_parquet(det_file.replace('detections_new_crop','detections_gaussian_2D1D'))
        print(f"Done with {im_file}, saved it at {det_file.replace('detections_new_crop','detections_gaussian_2D1D')}")



im_files = glob.glob('//tungstenfs/scratch/ggiorget/nessim/microscopy_data/2_color_contact_dynamics/full_dataset/*.nd2')

det_files = ['/tungstenfs/scratch/ggiorget/nessim/2_color_imaging/localization_precision_estimation/runs/run_full_dataset/detections_new_crop'+'/detections_'+ f.split('/')[-1].replace('.nd2','_cxy_9_cz_7.csv') for f in im_files]

if __name__ == '__main__':
    for files in zip(im_files,det_files):
        print(files)
        process_file(files)
    # #     break
    # with ThreadPoolExecutor(max_workers=20) as executor:
    #     executor.map(process_file,zip(im_files,det_files))

    # # main()
    