import numpy as np
import pandas as pd
from multiprocessing import Pool,Lock
from skimage.morphology import ball
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
import dask.array as da
from concurrent.futures import ProcessPoolExecutor, as_completed
import cProfile
import nd2


# function to filter the image

def format(im):
     im = white_tophat(lowpass(im,1),footprint=np.expand_dims(disk(2),axis=0))
     return im

def max_filter(raw_im):
    footprint=ball(7)
    max_filter=maximum_filter(raw_im,footprint=footprint)
    return max_filter

# function to perform the detections

def max5_detection(raw_im: np.ndarray,frame: int,max_filter_image:np.array,labels:np.array,original_im:np.ndarray,crop_size_xy:int = 4,crop_size_z:int = 4) -> pd.DataFrame:
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
    snr_o = []
    for l in np.unique(labels):
        if l == 0:
            continue
        else:
            im_masked_max = max_filter_image * (labels == l)

            raw_im_masked_ = raw_im * (labels == l)
            original_im_masked = original_im * (labels == l)

            raw_im_masked = raw_im_masked_*(raw_im_masked_ == im_masked_max)

            new_im_flat = raw_im_masked.flatten()
            original_im_masked_flat = original_im_masked.flatten()

            flat_index = np.argsort(new_im_flat)[-5:]
            intensity = new_im_flat[flat_index]

            intensity_o = original_im_masked_flat[flat_index]

            back_d = np.mean(raw_im_masked[raw_im_masked>0])

            back = np.mean(raw_im_masked_[raw_im_masked_>0])

            back_or = np.mean(original_im_masked[original_im_masked>0])

            snr_d.extend([i/back_d for i in intensity])
            snr.extend([i/back for i in intensity])
            snr_o.extend([i/back_or for i in intensity_o])

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
    snr_o = np.array(snr_o)[~near_edge]
    z,y,x = pos.T

    
    intensity = [original_im[z,y,x] for (z,y,x) in zip(z,y,x)]


    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print("No spots detected")
        return pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','channel','intensity','pixel_sum','label','snr_tophat','snr_dilated'])

    df_loc = pd.DataFrame([x,y,z]).T
    df_loc.columns = ['x','y','z']


    try:
        detections_refined= tp.refine.refine_com(raw_image = raw_im, image= raw_im, radius= [crop_size_z//2,crop_size_xy//2,crop_size_xy//2], coords = df_loc, max_iterations=1,
        engine='python', shift_thresh=0.6, characterize=True,
        pos_columns=['z','y','x'])
    except ValueError:
        detections_refined = df_loc

    df_loc[['z_fitted_refined','x_fitted_refined','y_fitted_refined']] = detections_refined[['z','y','x']]

    df_loc['frame'] = frame
    df_loc['intensity'] = intensity
    df_loc['pixel_sum'] = n_pixels
    df_loc['label'] = labs
    df_loc['snr_tophat'] = snr
    df_loc['snr_dilated'] = snr_d
    df_loc['snr_original'] = snr_o

    return df_loc

# function to load and save the data

def processing(input_path, output_file_path, frame, labels, crop_xy, crop_z):
    with lock:
        # Check if the output file already exists
        if os.path.exists(output_file_path):
            return None
    # Proceed with processing outside the lock to allow parallel computation
    im = nd2.imread(input_path, dask=True)
    labels = da.from_zarr(labels, component='0/')

    im = im[frame,:,...].compute()
    labels = labels[frame,...].compute()

    im_filtered = format(im)  # Assuming format is a predefined function
    max_filter_image = max_filter(im_filtered)  # Assuming max_filter is a predefined function

    # Assuming max5_detection is a predefined function that returns a DataFrame
    spot_df = max5_detection(raw_im=im_filtered, frame=frame, max_filter_image=max_filter_image, labels=labels, crop_size_xy=crop_xy, crop_size_z=crop_z,original_im=im)

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
        input_path, output_file_path, frame, labels, crop_xy, crop_z = args
        # Process each item in the chunk
        result = processing(input_path, output_file_path, frame, labels, crop_xy, crop_z)
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

    args = parser.parse_args()

    input_path = args.input_image
    # dask.config.set(scheduler='threads', num_workers=1)

    threads = args.threads
    
    im = nd2.imread(input_path, dask=True)

    output_path = (args.output_file).strip('.csv') 
    
    labels = args.labels

    n_frames = im.shape[0]

    tasks = [
    (input_path,f"{output_path}_{frame}.csv", frame, labels, args.crop_size_xy, args.crop_size_z)
    for frame in range(n_frames)]

    chunk_size =max(1, len(tasks) // (threads * 3))

    with Pool(initializer=init_process, initargs=(lock,),processes=threads) as pool:
        task_chunks = list(chunkify(tasks, chunk_size))
        _ = list(tqdm.tqdm(pool.imap(processing_chunk, task_chunks), total=len(task_chunks)))

    # Parallel file processing
    files_to_process = [os.path.join('/'.join(output_path.split('/')[:-1]), f) for f in os.listdir('/'.join(output_path.split('/')[:-1])) if f.endswith('.csv') and output_path.split('/')[-1] in f and 'wrong_intensity' not in f]
    with ProcessPoolExecutor(max_workers=threads) as executor:
        df_futures = [executor.submit(read_and_concatenate_csv, file) for file in files_to_process]
    
    df_final = pd.concat([future.result() for future in tqdm.tqdm(as_completed(df_futures), total=len(df_futures))])
    
    # Clean up files
    for file in files_to_process:
        os.remove(file)
    
    df_final.to_parquet(output_path+'.csv', index=False)

if __name__ == '__main__':
    main()
