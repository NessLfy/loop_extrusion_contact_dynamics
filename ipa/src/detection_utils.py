import numpy as np
import pandas as pd
from multiprocessing import Pool
from skimage.morphology import extrema
import os
from functools import partial
from localization_utils import gauss_single_spot,gauss_single_spot_2d


def hmax_3D(raw_im: np.ndarray,frame: int,sd: float,n:int = 4,thresh: float = 0.5,threads:int = 10) -> pd.DataFrame:
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
    #detect the spots
    im_mask = extrema.h_maxima(raw_im[frame],h=n*int(sd))

    # extract the points and fit gaussian
    z,y,x = np.nonzero(im_mask)
    # z = z*0.3
    # y = y*0.160
    # x = x*0.160
    
    # remove duplicates
    array = np.array([z,y,x]).T
    array = set([tuple(i) for i in array])
    z,y,x = zip(*array)

    k = [(raw_im[frame],y[i],x[i],z[i]) for i in range(len(x))]
    os.nice(19)
    with Pool(processes=threads) as p:
       x_s,y_s,z_s,sdx_fit,sdy_fit,sdz_fit = zip(*(p.starmap(gauss_single_spot,k)))
    os.nice(0)
    # x_s,y_s,z_s,sdx_fit,sdy_fit,sdz_fit,sd1,sd2,sd3 = x,y,z,np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x))

    # create a dataframe with sub pixel localization
    df_loc = pd.DataFrame([x_s,y_s,z_s,sdx_fit,sdy_fit,sdz_fit]).T
    df_loc.rename(columns={0:'x',1:'y',2:'z',3:'sd_fit_x',4:'sd_fit_y',5:'sd_fit_z'},inplace=True)
    df_loc['frame'] = [frame] * len(df_loc)

    # filter the dataframe based on the gaussian fit

    df_loc_filtered = df_loc.query(f'sd_fit_x <{thresh} and sd_fit_y <{thresh} and sd_fit_z < {thresh}') #remove the bad fit 
    df_loc_filtered = df_loc_filtered[df_loc_filtered.sd_fit_x.values != 0]    #remove the points that were not fitted
    # df_loc_filtered = df_loc


    return df_loc_filtered

def hmax_detection_fast(raw_im:np.array,frame:int,sd:float,n:int = 2,thresh:float = 0.5,threads:int=5) -> pd.DataFrame:
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
    
    #detect the spots
    im_mask = extrema.h_maxima(raw_im[frame],n*int(sd))

    # extract the points and fit gaussian

    y,x = np.nonzero(im_mask)  # coordinates of every ones

    # remove duplicates
    array = np.array([y,x]).T
    array = set([tuple(i) for i in array])
    y,x = zip(*array)

    partial_fit = partial(gauss_single_spot_2d,raw_im[frame])
    
    k = [(x[i],y[i]) for i in range(len(x))]

    with Pool(threads) as p:
        x_s,y_s,sdx_fit,sdy_fit,amp = zip(*(p.starmap(partial_fit,k)))
        
    # create a dataframe with sub pixel localization

    df_loc = pd.DataFrame([x_s,y_s,sdx_fit,sdy_fit,amp]).T
    df_loc.rename(columns={0:'x',1:'y',2:'sd_fit_x',3:'sd_fit_y',4:'amplitude'},inplace=True)
    df_loc['frame'] = [frame] * len(df_loc)
    df_loc
    
    # filter the dataframe based on the gaussian fit

    df_loc_filtered = df_loc.query(f'sd_fit_x <{thresh} and sd_fit_y <{thresh}') #remove the bad fit 
    df_loc_filtered = df_loc_filtered[df_loc_filtered.sd_fit_x.values != 0]    #remove the points that were not fitted
    df_loc_filtered

    return df_loc_filtered
