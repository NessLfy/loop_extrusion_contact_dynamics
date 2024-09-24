import numpy as np
import pandas as pd
from multiprocessing import Pool
from skimage.morphology import ball, dilation
import ipa.src.snakemake_utils as snk
import nd2
from skimage.morphology import disk,white_tophat
from trackpy.preprocessing import lowpass
from itertools import starmap
import sys
sys.path.append("/tungstenfs/scratch/ggiorget/nessim/2_color_imaging/localization_precision_estimation/ipa/src")
from localization_utils import locate_com,gauss_single_spot_2d_1d
import trackpy as tp
import argparse
import tqdm
import os
from scipy.ndimage import maximum_filter
import dask.array as da
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock

# function to filter the image

def format(im):
     im = white_tophat(lowpass(im,1),footprint=np.expand_dims(disk(2),axis=0))
     return im

def max_filter(raw_im):
    footprint=ball(7)
    max_filter=maximum_filter(raw_im,footprint=footprint)
    return max_filter

# function to perform the detections

def max5_detection(filtered_image: np.ndarray,raw_image:np.array,frame: int,channel:int,max_filter_image:np.array,labels:np.array,crop_size_xy:int = 4,crop_size_z:int = 4,method:str='com') -> pd.DataFrame:
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
    snr_o = []
    for l in np.unique(labels):
        if l == 0:
            continue
        else:
            im_masked_max = max_filter_image * (labels == l)

            filtered_image_masked = filtered_image * (labels == l)

            raw_im_masked_ = raw_image * (labels == l)

            filtered_image_masked = filtered_image_masked*(filtered_image_masked == im_masked_max)

            new_im_flat = filtered_image_masked.flatten()
            flat_index = np.argsort(new_im_flat)[-5:]

            intensity = new_im_flat[flat_index]

            intensity_o = raw_image.flatten()[flat_index]

            back_d = np.mean(raw_im_masked_[raw_im_masked_>0])
            back = np.mean(filtered_image_masked[filtered_image_masked>0])

            snr_o.extend([i/back_d for i in intensity_o])
            snr.extend([i/back for i in intensity])

            max_coords_2.extend([np.unravel_index(f, filtered_image_masked.shape) for f in flat_index])

            n_pixels.extend([np.sum(labels == l)]*5)
            labs.extend([l]*5)

    try:
        z,y,x = np.array(max_coords_2).T
    except ValueError as e:
        print(f'The following error has occured when trying to extract segmented pixels {e}')

    # remove duplicates
    pos=np.vstack((z,y,x)).T
    margin=np.array((crop_size_z//2,crop_size_xy//2,crop_size_xy//2))
    shape=filtered_image.shape
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]
    labs = np.array(labs)[~near_edge]
    n_pixels = np.array(n_pixels)[~near_edge]
    snr = np.array(snr)[~near_edge]
    snr_o = np.array(snr_o)[~near_edge]
    z,y,x = pos.T

    intensity = [filtered_image[z,y,x] for (z,y,x) in zip(z,y,x)]

    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print(np.array(max_coords_2).T)
        print("No spots detected")
        return pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','channel','intensity','pixel_sum','label','snr_tophat','snr_dilated'])

    df_loc = pd.DataFrame([x,y,z]).T
    df_loc.columns = ['x','y','z']
    # df_loc.rename(columns={0:'x',1:'y',2:'z',3:'x_fitted',4:'y_fitted',5:'z_fitted'},inplace=True)
    if method.lower() == 'com':
        try:
            detections_refined= tp.refine.refine_com(raw_image = filtered_image, image= filtered_image, radius= [crop_size_z//2,crop_size_xy//2,crop_size_xy//2], coords = df_loc, max_iterations=1,
            engine='python', shift_thresh=0.6, characterize=True,
            pos_columns=['z','y','x']) # order matters
        except ValueError:
            detections_refined = df_loc

        df_loc[['z_fitted_refined','x_fitted_refined','y_fitted_refined']] = detections_refined[['z','x','y']]

    elif method.lower() == 'gaussian':
        k = [(filtered_image,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z) for i in range(len(pos))]
        x_s,y_s,z_s,sx,sy,sz= zip(*(starmap(gauss_single_spot_2d_1d,k)))
        df_loc = pd.DataFrame([x,y,z,x_s,y_s,z_s,sx,sy,sz]).T
        df_loc.columns=['x','y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined',"sigma_x","sigma_y","sigma_z"]
    else:
        print('Please provide a valid method for fitting the spots')


    df_loc['frame'] = frame
    df_loc['channel'] = channel
    df_loc['intensity'] = intensity
    df_loc['pixel_sum'] = n_pixels
    df_loc['label'] = labs
    df_loc['snr_tophat'] = snr
    df_loc['snr_original'] = snr_o

    return df_loc


# function to load and save the data

def processing(input_path, output_file_path, frame, channel, labels, crop_xy, crop_z,method):
    with lock:
        # Check if the output file already exists
        if os.path.exists(output_file_path):
            return None
    # Proceed with processing outside the lock to allow parallel computation
    im = nd2.imread(input_path, dask=True)

    if len(im.shape) != 5:
        im = im.reshape((im.shape[0]//15,15,2,im.shape[-2],im.shape[-1]))
    

    labels = da.from_zarr(labels, component='0/')

    im = im[frame,:,channel,...].compute()
    labels = labels[frame,...].compute()

    im_filtered = format(im)  # Assuming format is a predefined function
    max_filter_image = max_filter(im_filtered)  # Assuming max_filter is a predefined function

    # Assuming max5_detection is a predefined function that returns a DataFrame
    spot_df = max5_detection(filtered_image=im_filtered, raw_image=im ,frame=frame, channel=channel, max_filter_image=max_filter_image, labels=labels, crop_size_xy=crop_xy, crop_size_z=crop_z,method=method)

    # Use the lock again for writing the file to ensure thread-safe I/O
    with lock:
        spot_df.to_csv(output_file_path, index=False)

def init_process(shared_lock):
    global lock
    lock = shared_lock

def processing_chunk(chunk):
    results = []
    for args in chunk:
        # Unpack the arguments
        input_path, output_file_path, frame, channel, labels, crop_xy, crop_z,method = args
        # Process each item in the chunk
        result = processing(input_path, output_file_path, frame, channel, labels, crop_xy, crop_z,method)
        results.append(result)
    return results

def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def read_and_concatenate_csv(files):
    df_temp = pd.read_csv(files)
    return df_temp

def main():
    lock = Lock()
    # Create the parser
    parser = argparse.ArgumentParser(description='Run detections')

    # Add the arguments
    parser.add_argument('--input_image', type=str, help='The input image file')
    parser.add_argument('--output_file', type=str, help='The output file')
    parser.add_argument('--threads', type=int, help='The number of threads')
    parser.add_argument('--labels', type=str, help='label_image')
    parser.add_argument('--crop_size_xy', type=int, help='crop_size_xy')
    parser.add_argument('--crop_size_z', type=int, help='crop_size_z')
    parser.add_argument('--method', type=str, help='the fitting method')

    args = parser.parse_args()

    input_path = args.input_image
    # dask.config.set(scheduler='threads', num_workers=1)

    threads = args.threads
    
    #im = nd2.imread(input_path, dask=True)

    output_path = (args.output_file).strip('.csv') 
    
    labels = args.labels

    n_frames = 4#int(np.shape(im)[0])

    tasks = [
    (input_path,f"{output_path}_{frame}_{channel}.csv", frame, channel, labels, args.crop_size_xy, args.crop_size_z, args.method)
    for frame in range(n_frames) for channel in range(2)]

    chunk_size =max(1, len(tasks) // (threads * 3))

    with Pool(initializer=init_process, initargs=(lock,),processes=threads) as pool:
        task_chunks = list(chunkify(tasks, chunk_size))
        _ = list(tqdm.tqdm(pool.imap(processing_chunk, task_chunks), total=len(task_chunks)))



    # Parallel file processing
    files_to_process = [os.path.join('/'.join(output_path.split('/')[:-1]), f) for f in os.listdir('/'.join(output_path.split('/')[:-1])) if f.endswith('.csv') and output_path.split('/')[-1] in f]
    with ProcessPoolExecutor(max_workers=threads) as executor:
        df_futures = [executor.submit(read_and_concatenate_csv, file) for file in files_to_process]
    
    df_final = pd.concat([future.result() for future in tqdm.tqdm(as_completed(df_futures), total=len(df_futures))])
    
    # Clean up files
    for file in files_to_process:
        os.remove(file)
    
    df_final.to_parquet(output_path+'.csv', index=False)

if __name__ == '__main__':
    main()