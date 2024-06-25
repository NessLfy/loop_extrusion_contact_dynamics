import numpy as np
import pandas as pd
from multiprocessing import Pool
from skimage.morphology import extrema
import os
from functools import partial
import sys
sys.path.append("/tungstenfs/scratch/ggiorget/nessim/2_color_imaging/localization_precision_estimation/ipa/src")
from localization_utils import gauss_single_spot,gauss_single_spot_2d,locate_com 
from itertools import starmap
import trackpy as tp

def hmax_3D(raw_im: np.ndarray,frame: int,sd: float,n:int = 4,thresh: float = 0.5,threads:int = 10,crop_size_xy:int = 4,crop_size_z:int = 4,fitting:bool = True,method:str = "gaussian") -> pd.DataFrame:
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
    pos=np.vstack((z,y,x)).T
    margin=np.array((crop_size_z//2,crop_size_xy//2,crop_size_xy//2))
    shape=raw_im[frame].shape
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]
    z,y,x = pos.T
    
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print("No spots detected")
        return pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','method'])
    
    if fitting == True:
        k = [(raw_im[frame],pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z) for i in range(len(pos))]
        if(method.lower()=="gaussian"):
            with Pool(processes=threads) as p:
                x_s,y_s,z_s,*_ = zip(*(p.starmap(gauss_single_spot,k)))
        elif(method.lower()=="com"):
            with Pool(processes=threads) as p:
                x_s,y_s,z_s= zip(*(p.starmap(locate_com,k)))
        
        else:
            raise Exception("Invalid method")
        
        df_loc = pd.DataFrame([x,y,z,x_s,y_s,z_s]).T
        df_loc.rename(columns={0:'x',1:'y',2:'z',3:'x_fitted',4:'y_fitted',5:'z_fitted'},inplace=True)
        df_loc['frame'] = frame
        df_loc['method'] = method

    else:
        df_loc = pd.DataFrame([x,y,z]).T
        df_loc.rename(columns={0:'x',1:'y',2:'z'},inplace=True)
        df_loc['frame'] = [frame] * len(df_loc)

    return df_loc

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

def fitting(raw_im: np.ndarray,frame: int,detected_spot: pd.DataFrame,thresh: float = 0.5,threads:int = 10) -> pd.DataFrame:
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
    x=detected_spot['x'].values
    y=detected_spot['y'].values
    z=detected_spot['z'].values
    
    # remove duplicates
    array = np.array([z,y,x]).T
    array = set([tuple(i) for i in array])
    z,y,x = zip(*array)
 
    k = [(raw_im[frame],y[i],x[i],z[i]) for i in range(len(x))]
    #os.nice(19)
    with Pool(processes=threads) as p:
        x_s,y_s,z_s,sdx_fit,sdy_fit,sdz_fit = zip(*(p.starmap(gauss_single_spot,k)))
    #os.nice(0)
    
    # create a dataframe with sub pixel localization
    df_loc = pd.DataFrame([x_s,y_s,z_s,sdx_fit,sdy_fit,sdz_fit]).T
    df_loc.rename(columns={0:'x',1:'y',2:'z',3:'sd_fit_x',4:'sd_fit_y',5:'sd_fit_z'},inplace=True)
    df_loc['frame'] = [frame] * len(df_loc)
 
    # filter the dataframe based on the gaussian fit
    df_loc_filtered = df_loc.query(f'sd_fit_x <{thresh} and sd_fit_y <{thresh} and sd_fit_z < {thresh}') #remove the bad fit
    df_loc_filtered = df_loc_filtered[df_loc_filtered.sd_fit_x.values != 0]    #remove the points that were not fitted
 
    return df_loc_filtered


def hmax_3D_dask(raw_im: np.ndarray,frame: int,channel:int,sd: float,n:int = 4,crop_size_xy:int = 4,crop_size_z:int = 4,chunk_location:tuple = (0,0,0,0,0)) -> pd.DataFrame:
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
    # print(np.shape(raw_im))
    # return None
    #detect the spots
    im_mask = extrema.h_maxima(raw_im,h=n*int(sd))
    #print(f"Frame {frame} channel {channel} spots detected: {np.shape(np.nonzero(im_mask))[1]}")

    if np.shape(np.nonzero(im_mask))[1] == 0:
        print("No spots detected")
        return np.empty((0,12))
    # extract the points and fit gaussian
    try:
        z,y,x = np.nonzero(im_mask)
    except ValueError as e:
        print(f'The following error has occured when trying to extract segmented pixels {e}')
        return np.empty((0,12))#pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','chunk_location'])
    
    # remove duplicates
    array = np.array([z,y,x]).T
    array = set([tuple(i) for i in array])
    z,y,x = zip(*array)
    pos=np.vstack((z,y,x)).T
    margin=np.array((crop_size_z//2,crop_size_xy//2,crop_size_xy//2))
    shape=raw_im.shape
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]
    z,y,x = pos.T
    
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print("No spots detected")
        return np.empty((0,12))#pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','chunk_location'])

    try:
        k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z) for i in range(len(pos))]
    except IndexError as e:
        print(f'The following error has occured when trying to create the arguments for the fitting {e}')
        return np.empty((0,12))#pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','chunk_location'])
    
    try:
        x_s,y_s,z_s= zip(*(starmap(locate_com,k)))
    except IndexError as e:
        print(f'An error occurred when trying to map the locate_com function: {e}')
        return np.empty((0,12))#pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','chunk_location'])
        
    df_loc = pd.DataFrame([x,y,z,x_s,y_s,z_s]).T
    df_loc.rename(columns={0:'x',1:'y',2:'z',3:'x_fitted',4:'y_fitted',5:'z_fitted'},inplace=True)

    try:
        detections_refined= tp.refine.refine_com(raw_image = raw_im, image= raw_im, radius= crop_size_xy//2, coords = df_loc, max_iterations=1,
        engine='python', shift_thresh=0.6, characterize=True,
        pos_columns=['z_fitted','y_fitted','x_fitted'])
    except ValueError:
        detections_refined = df_loc

    df_loc[['z_fitted_refined','x_fitted_refined','y_fitted_refined']] = detections_refined[['z_fitted','x_fitted','y_fitted']]

    df_loc['frame'] = frame
    df_loc['channel'] = channel
    df_loc['chunk_location'] = [chunk_location] * len(df_loc)
    return df_loc.values