import numpy as np
import pandas as pd
from skimage.morphology import disk,ball,white_tophat
from scipy.ndimage import maximum_filter
import logging
from datetime import datetime
from trackpy.preprocessing import lowpass
from skimage.feature import blob_log

def get_loc(im:np.array,frame:int,mins:float,maxs:float,thresh:float,nums:int=10 )-> pd.DataFrame:

    """Function to return localizations from a laptrack detection

    Args:
        im (np.array): input image
        mins (float): minimum sigma used for the detection see skimage.feature blob_log for more details
        maxs (float): maximum sigma used for the detection see skimage.feature blob_log for more details
        nums (float): number of sigma tested for the detection see skimage.feature blob_log for more details
        thresh (float): relative threshold used for the detection see skimage.feature blob_log for more details. Defaults to 10.

    Returns:
        pd.DataFrame: dataframe of all the localizations (gaussian fitted)
    """
    
    ima = im[frame].copy()
    
    df = lap(ima,mins=mins,maxs=maxs,nums=nums,thresh=thresh)
    x_loc =[]
    y_loc =[]
    for i in df.iloc:
        x_loc.append(i.x)
        y_loc.append(i.y)

    df['x'] = x_loc
    df['y'] = y_loc

    return df

def lap(im:np.array,mins:float,maxs:float,thresh:float,nums:int=10) -> pd.DataFrame: 
    """Function to compute laptrack spot detection

    Args:
        im (np.array): input image
        mins (float): minimum sigma used for the detection see skimage.feature blob_log for more details
        maxs (float): maximum sigma used for the detection see skimage.feature blob_log for more details
        nums (float): number of sigma tested for the detection see skimage.feature blob_log for more details; Default to
        thresh (float): relative threshold used for the detection see skimage.feature blob_log for more details

    Returns:
        pd.DataFrame: df of all the detections
    """
    images = im

    _spots = blob_log(images.astype(float), min_sigma=mins, max_sigma=maxs, num_sigma=nums,threshold_rel=thresh)
    rad = [np.sqrt(2*_spots[x][-1]) for x in range(len(_spots))]
    df = pd.DataFrame(_spots, columns=["y", "x", "sigma"])
    
    df['radius'] = rad
    
    df = df[df.radius > 1.2]
    
    df.reset_index(drop=True,inplace=True)
    return df

def compute_h_param(im:np.array,frame:int,mins:float = 1.974 ,maxs:float = 3.0 ,thresh:float = 0.884) -> float:

    # Compute LoG with very high threhsold 

    df = get_loc(im,frame,mins,maxs,thresh)

    # if there is no spot be a bit more gentle with the parameters
    if len(df) == 0:
        return 0
    else:
        #compute the sd of the detected spots

        _,sd,_ = heatmap_detection(im,frame=frame,df=df,name='sd')

        # compute the mean sd across the image 

        mean_sd = np.mean(sd)

        return mean_sd

def heatmap_detection(raw_im:np.array,frame:int,df:pd.DataFrame,name:str)-> tuple:
    """Create a heatmap to compute the threshold for h-max detection

    Args:
        raw_im (np.array): the raw image to segment
        frame (int): the frame to segment
        df (pd.DataFrame): detections (x,y) on the raw image to be able to compute the intensity profiles of the detected spots
        name (str): either 'med' or 'sd'. Whether you want to display the median pixel intensity value around the spots or the sd. In any case it returns the values of the 2

    Returns:
        tuple: heatmap: a n-dimentional array (shape of the image) with either the median pixel intensity value for each bins or the sd of each bin, the median pixel intensity value for all detected spots provided, the sd of the intensity of every provided spots
    """
    # create image with extended boarders to be able to take bbox

    im = np.pad(raw_im[frame],4).T

    # create the same for the mask

    im_mask = np.pad(np.ones_like(raw_im[frame],dtype=int),4)

    spot = []
    for i in df.iloc:
        x,y = int(i.x+3),int(i.y+3)
        patch = im[x-2:x+3,y-2:y+3]
        #print(patch)
        #break
        #patch_mask = im_mask[x-4:x+5,y-4:y+5]*disk(4)  # get only a disk in the bbox
        #masked_patch = patch[patch_mask]
        #print(masked_patch)
        #break
        spot.append(patch.ravel()) # get a 1d list of all bbox 

    med = np.median(spot,axis=1) # median of bbox where there are spots
    sd = np.std(spot,axis=1) # sd of bbox where there are spots

    df['med'] = med
    df['sd'] = sd

    df_heat = df[['x','y','med','sd']].copy(deep=True)

    x = []
    y = []
    for i,j in zip(df['x'].values,df['y'].values):
        try:
            x.append(i//32)
            y.append(j//32)
        except RuntimeWarning:
            x.append(0)
            y.append(0)

    df_heat['x'] = x#df['x'].values//32
    df_heat['y'] = y#df_heat['y'].values//32 # bin the image to categories to be able to see better the spots 

    heatmap = []
    for i in range(int(max(df_heat.y.values))):
        list_heat_row =[]
        for j in range(int(max(df_heat.x.values))):
            try:
                list_heat_row.append(np.mean(df_heat[(df_heat.x == j) & (df_heat.y == i)][name].values)) # average the region binned (row by row)
            except RuntimeWarning:
                list_heat_row.append(0)
                
        heatmap.append(list_heat_row)

    heatmap = np.array(heatmap)

    return heatmap,sd,med

def _create_logger(name: str) -> logging.Logger:
    """
    Create logger which logs to <timestamp>-<name>.log inside the current
    working directory.

    Args: 
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Logger
    """
    logger = logging.Logger(name.capitalize())
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    handler = logging.FileHandler(f"{now}-{name}.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def max_filter(raw_im):
    '''
    Function to perform a maximum filter on the image

    Args:
    raw_im: np.array, the image to filter

    Returns:
    max_filter: np.array, the filtered image

    This function performs a maximum filter on the image using a ball of radius 7 as footprint
    '''

    footprint=ball(7)
    max_filter=maximum_filter(raw_im,footprint=footprint)

    return max_filter

def format(im):
    '''
    Function to filter the image

    Args:
    im: np.array, the image to filter

    Returns:
    im: np.array, the filtered image

    This function filters the image using a white tophat filter with a disk of radius 2 as footprint and a lowpass filter with a sigma of 1
    '''
    im = white_tophat(lowpass(im,1),footprint=np.expand_dims(disk(2),axis=0))
    return im

def format_gaussian(im):
    im = lowpass(im,1)
    return im