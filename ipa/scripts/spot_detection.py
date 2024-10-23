import numpy as np
import pandas as pd
from multiprocessing import Pool,Lock
import dask.array as da
from detection_utils import max5_detection
from preprocessing_utils import max_filter,format,format_gaussian
import argparse
import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import nd2

# function to load and save the data

def processing(input_path, output_file_path, frame, channel, labels, crop_xy, crop_z,method):
    """
    Function to process the input data and save the output to a file

    Args:
        input_path (str): The input image file
        output_file_path (str): The output file
        frame (int): The frame to process
        channel (int): The channel to process
        labels (str): The label image
        crop_xy (int): The crop size in the xy plane
        crop_z (int): The crop size in the z plane

    Returns:
        None

    This function processes the input data and saves the output to a file. The function first checks if the output file already exists and returns if it does.
    The function then loads the input image and the labels and processes the image using the functions format and max_filter. 
    The function then detects the spots using the function max5_detection and saves the output to a file.
    The function uses a lock to ensure thread-safe I/O operations.
    """

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
    #gaussian_im = format_gaussian(im)
    # Assuming max5_detection is a predefined function that returns a DataFrame
    spot_df = max5_detection(raw_im=im, frame=frame, channel=channel, max_filter_image=max_filter_image, labels=labels, crop_size_xy=crop_xy, crop_size_z=crop_z,filtered_image=im_filtered,method=method)

    # Use the lock again for writing the file to ensure thread-safe I/O
    with lock:
        spot_df.to_csv(output_file_path, index=False)

def init_process(shared_lock):
    '''
    Function to initialize the lock for the parallel processing

    Args:
    shared_lock (Lock): The lock to use for the parallel processing

    Returns:
    None
    '''
    global lock
    lock = shared_lock

def processing_chunk(chunk):
    '''
    Function to process a chunk of data

    Args:
    chunk (list): The chunk of data to process

    Returns:
    list: The results of the processing

    This function processes a chunk of data in parallel using the processing function.
    The function returns the results of the processing.
    '''
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
    '''
    Function to read and concatenate the csv files

    Args:
    files (str): The file to read and concatenate

    Returns:
    pd.DataFrame: The concatenated dataframe

    This function reads and concatenates the csv files and returns the concatenated dataframe.
    '''

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
    parser.add_argument('--method', type=str, help='method for fitting')

    args = parser.parse_args()

    input_path = args.input_image
    # dask.config.set(scheduler='threads', num_workers=1)

    threads = args.threads
    method = args.method
    
    # im = da.from_zarr(input_path, component='0/')
    
    im = nd2.imread(input_path, dask=True)
    if len(np.shape(im)) == 4:
        im = im.reshape(im.shape[0]//15,15,2,im.shape[-2],im.shape[-1])

    output_path = (args.output_file).strip('.csv') 
    
    labels = args.labels

    n_frames = np.shape(im)[0]

    tasks = [
    (input_path,f"{output_path}_{frame}_{channel}.csv", frame, channel, labels, args.crop_size_xy, args.crop_size_z,method)
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
