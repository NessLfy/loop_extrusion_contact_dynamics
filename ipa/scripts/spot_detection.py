import numpy as np
import pandas as pd
from multiprocessing import Pool,Lock
from skimage.morphology import ball
import dask.array as da
from skimage.morphology import disk,white_tophat
from trackpy.preprocessing import lowpass
from itertools import starmap
from localization_utils import find_start_end, gauss_single_spot_2d_1d
import trackpy as tp
import argparse
import tqdm
import os
from scipy.ndimage import maximum_filter
import dask.array as da
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def process_labels(l,labels,raw_im,max_filter_image,filtered_image):

    im_masked_max = max_filter_image * (labels == l)

    filtered_image_masked = filtered_image * (labels == l)

    raw_im_masked_ = raw_im * (labels == l)

    filtered_image_masked_ = filtered_image_masked*(filtered_image_masked == im_masked_max)

    new_im_flat = filtered_image_masked_.flatten()
    flat_index = np.argsort(new_im_flat)[-5:]

    intensity = new_im_flat[flat_index]

    intensity_o = raw_im.flatten()[flat_index]

    back_d = np.mean(raw_im_masked_[raw_im_masked_>0])
    back = np.mean(filtered_image_masked[filtered_image_masked>0])

    snr_o = [i/back_d for i in intensity_o]
    snr  = [i/back for i in intensity]

    max_coords_2 = [np.unravel_index(f, raw_im_masked_.shape) for f in flat_index]

    n_pixels = [np.sum(labels == l)]*5
    labs = [l]*5

    return max_coords_2,n_pixels,snr,snr_o,labs

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

def max5_detection(raw_im: np.ndarray,filtered_image:np.ndarray,frame: int,channel:int,max_filter_image:np.array,labels:np.array,crop_size_xy:int = 4,crop_size_z:int = 4,method:str = "gauss") -> pd.DataFrame:
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

    unique_labs = np.unique(labels)
    unique_labs = unique_labs[unique_labs != 0]

    k = [(l,labels,raw_im,max_filter_image,filtered_image) for l in unique_labs if l != 0]

    max_coords_2,n_pixels,snr,snr_o,labs = zip(*list(starmap(process_labels,k)))

    labs = np.concatenate(labs)
    n_pixels = np.concatenate(n_pixels)
    snr = np.concatenate(snr)
    snr_o = np.concatenate(snr_o)

    try:
        z,y,x = np.concatenate(max_coords_2).T
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
    snr_o = np.array(snr_o)[~near_edge]
    z,y,x = pos.T

    intensity = [raw_im[z,y,x] for (z,y,x) in zip(z,y,x)]

    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print("No spots detected")
        return pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','channel','intensity','pixel_sum','label','snr_tophat','snr_original'])


    df_loc = pd.DataFrame([x,y,z]).T
    df_loc.columns = ['x','y','z']

    if method == 'com':
        fitted_2d = []
        for i,row in df_loc.iterrows():
            df_temp = df_loc[(df_loc.x == row.x)&(df_loc.y == row.y)&(df_loc.z == row.z)]

            if len(df_temp) == 1:
                detections_refined= tp.refine.refine_com(raw_image = filtered_image[int(row.z),...], image= raw_im[int(row.z),...], radius= [crop_size_xy//2,crop_size_xy//2], coords = df_temp, max_iterations=1,
                engine='python', shift_thresh=0.6, characterize=False,
                pos_columns=['y','x'])
                fitted_2d.append(detections_refined[['x','y']])
            else:
                print("Multiple detections at the same location", df_temp)
                df_temp = df_temp.iloc[0:1]
                detections_refined= tp.refine.refine_com(raw_image = filtered_image[int(row.z),...], image= raw_im[int(row.z),...], radius= [crop_size_xy//2,crop_size_xy//2], coords = df_temp, max_iterations=1,
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

    elif method == 'gauss':
        k = [(filtered_image,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z,raw_im) for i in range(len(pos))]
        x_s,y_s,z_s,sx,sy,sz,max_spot,mean_back,std_back,max_spot_tophat,mean_back_tophat,std_back_tophat = zip(*(starmap(gauss_single_spot_2d_1d,k)))
        df_loc = pd.DataFrame([x,y,z,x_s,y_s,z_s,sx,sy,sz,max_spot,mean_back,std_back ,max_spot_tophat,mean_back_tophat,std_back_tophat]).T
        df_loc.columns=['x','y','z',
                        'x_fitted_refined','y_fitted_refined','z_fitted_refined',
                        "sigma_x","sigma_y","sigma_z",
                        "max_original","mean_back_original","std_back_original",
                        "max_tophat","mean_back_tophat","std_back_tophat"]

    df_loc['frame'] = frame
    df_loc['channel'] = channel
    df_loc['intensity'] = intensity
    df_loc['pixel_sum'] = n_pixels
    df_loc['label'] = labs
    df_loc['snr_tophat'] = snr
    df_loc['snr_original'] = snr_o

    return df_loc

# function to load and save the data

def processing(input_path, output_file_path, frame, channel, labels, crop_xy, crop_z):
    with lock:
        # Check if the output file already exists
        if os.path.exists(output_file_path):
            return None
    # Proceed with processing outside the lock to allow parallel computation
    # im = da.from_zarr(input_path, component='0/')
    im = nd2.imread(input_path, dask=True)
    if len(np.shape(im)) == 4:
        im = im.reshape(im.shape[0]//15,15,2,im.shape[-2],im.shape[-1])

    # labels = zarr.open_group(labels)
    labels = da.from_zarr(labels, component='0/')

    # im = im[frame,:,...].compute()
    im = im[frame,:,channel,...].compute()
    labels = labels[frame,...].compute()

    im_filtered = format(im)  # Assuming format is a predefined function
    max_filter_image = max_filter(im_filtered)  # Assuming max_filter is a predefined function

    # Assuming max5_detection is a predefined function that returns a DataFrame
    spot_df = max5_detection(raw_im=im, frame=frame, channel=channel, max_filter_image=max_filter_image, labels=labels, crop_size_xy=crop_xy, crop_size_z=crop_z,filtered_image=im_filtered)

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
        input_path, output_file_path, frame, channel, labels, crop_xy, crop_z = args
        # Process each item in the chunk
        result = processing(input_path, output_file_path, frame, channel, labels, crop_xy, crop_z)
        results.append(result)
    return results

def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def read_and_concatenate_csv(files):
    try:
        df_temp = pd.read_csv(files)
    except pd.errors.EmptyDataError:
        print(f"Empty file: {files}")
        return pd.DataFrame()
    
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
    
    # im = da.from_zarr(input_path, component='0/')
    im = nd2.imread(input_path, dask=True)
    if len(np.shape(im)) == 4:
        im = im.reshape(im.shape[0]//15,15,2,im.shape[-2],im.shape[-1])

    output_path = (args.output_file).strip('.csv') 
    
    labels = args.labels

    n_frames = np.shape(im)[0]

    tasks = [
    (input_path,f"{output_path}_{frame}_{channel}.csv", frame, channel, labels, args.crop_size_xy, args.crop_size_z)
    for frame in range(n_frames) for channel in range(2)]

    # channel = 0

    # tasks = [
    # (input_path,f"{output_path}_{frame}.csv", frame, channel,labels, args.crop_size_xy, args.crop_size_z)
    # for frame in range(n_frames)]


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
