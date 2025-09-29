import numpy as np
import pandas as pd
import nd2
from glob import glob
from multiprocessing import Pool
import sys
sys.path.append('./ipa/src/')
import preprocessing_utils as pre
import nd2
from skimage.morphology import disk
import zarr
from scipy.ndimage import maximum_filter
import argparse
from tqdm import tqdm
import dask
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings("ignore")

def create_footprint(d_xy,d_z):
    footprint=np.zeros((d_z,disk(d_xy//2).shape[0],disk(d_xy//2).shape[0]))
    for i in range(len(footprint)):
        footprint[i]=disk(d_xy//2)
    return footprint

def preprocess_image(args):
    image_path, frame, channel,l_cell = args
    
    with nd2.ND2File(image_path) as f:
        im = f.to_dask()
        im_channel = im[frame, :, channel, ...].compute()
        im_segmented = im_channel * l_cell
        im_c1_tophat = pre.format(im_segmented)
    return im_c1_tophat

dask.config.set(scheduler='single-threaded', num_workers=1)


def process_label(args):
    labels_path, id_track, frame, new_label, im_path, footprint, channel = args

    label_tracks = zarr.open(labels_path)[0]

    l = id_track
    l_cell = np.where(label_tracks[frame] == l, 1, 0)

    im_segmented = preprocess_image((im_path, frame, channel, l_cell))

    threshold = np.percentile(im_segmented, 99)
    im_segmented = im_segmented * (im_segmented > threshold)
    im_max = maximum_filter(im_segmented, footprint=footprint)
    im_max = im_max * (im_segmented == im_max)

    flat_index = np.argsort(im_max.flatten())[-2:]
    max_coords = np.array([np.unravel_index(f, im_max.shape) for f in flat_index])

    d_temp_c1 = np.sqrt(np.sum(((max_coords[0] - max_coords[1]) * (0.3, 0.13, 0.13)) ** 2))
    flat_index = np.argsort(im_segmented.flatten())[-2:]

    max_coords = np.array([np.unravel_index(f, im_segmented.shape) for f in flat_index])
    d_temp_c1_non_max = np.sqrt(np.sum(((max_coords[0] - max_coords[1]) * (0.3, 0.13, 0.13)) ** 2))

    return labels_path, frame, id_track, new_label, d_temp_c1, d_temp_c1_non_max, channel

def get_result(future):
    return future.result()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Path to the nd2 file")
    parser.add_argument("--file_label", type=str, help="Path to the label")
    parser.add_argument("--tracks", type=str, help="Path to the tracks")
    parser.add_argument("--cx", type=int, help="Size of the footprint in x",default=3)
    parser.add_argument("--cz", type=int, help="Size of the footprint in z",default=3)
    parser.add_argument("--threads", type=int, help="Number of threads",default=5)
    parser.add_argument("--output_file", type=str, help="Output file",default="distance_max2.csv")

    args = parser.parse_args()

    tracks = pd.read_parquet(args.tracks)
    

    if len(tracks)==0:
        print('No tracks found')
        return pd.DataFrame().to_parquet(args.output_file, index=False)

    track_id = tracks[['label','frame','new_label']].values

    image_path = args.image_path

    cx = args.cx
    cz = args.cz
    threads = args.threads
    output= args.output_file

    #create the list of tasks

    footprint = create_footprint(cx,cz)

    # Preprocess the image once for each unique frame and channel combination using Pool
    # unique_frames_channels = {(frame, channel) for _, frame, _ in track_id for channel in range(2)}
    # preprocessed_images = {}

    # with Pool(threads) as pool:
    #     results = list(tqdm(pool.imap(preprocess_image, [(image_path, frame, channel) for frame, channel in unique_frames_channels]), total=len(unique_frames_channels)))
    #     preprocessed_images = dict(results)

    # Create the tasks
    # tasks = [(args.file_label, int(id), int(frame), new_label, footprint, int(channel)) for (id, frame, new_label) in track_id for channel in range(2)]

    tasks = [(args.file_label, int(id), int(frame), new_label, image_path , footprint, int(channel)) for (id, frame, new_label) in track_id for channel in range(2)]

    # Use ThreadPoolExecutor to process the tasks
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_label, task) for task in tasks]
        distances = list(tqdm(map(get_result, futures), total=len(tasks)))


    dfs=[]
    for i in range(len(distances)):
        dfs.append(pd.DataFrame(distances[i]).T)

    df=pd.concat(dfs)
    df.columns=["filename","frame","label","new_label","d","d_non_max",'channel']

    df.to_parquet(output, index=False)

if __name__ == '__main__':
    
    main()