import numpy as np
import pandas as pd
from multiprocessing import Pool
from skimage.morphology import extrema,ball,dilation
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


def hmax_3D_dask(raw_im: np.ndarray,frame: int,channel:int,n_labels:int,crop_size_xy:int = 4,crop_size_z:int = 4,chunk_location:tuple = (0,0,0,0,0)) -> pd.DataFrame:
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

    # find the best n and sd

    h = np.mean(raw_im) + 2*np.std(raw_im,dtype=np.float64) # to avoid overflow errors

    part_func = partial(compute_n_param,raw_im=raw_im,h=h,crop_size_xy=crop_size_xy,crop_size_z=crop_size_z)

    # best_n = list(zip(*map(part_func,[x for x in np.arange(1,10,0.5)])))
    n_cond = 1.0 
    print(f'h value: {h}, n_cond: {n_cond}')
    n_detect_best = part_func(n_cond)

    # n_detect_best = part_func(n)
    if n_detect_best < n_labels:
        while n_detect_best < n_labels:
            n_cond = n_cond - 0.5
            n_detect_best = part_func(n_cond)
            if n_cond == 0.:
                break

    elif n_detect_best > n_labels:
        while n_detect_best > n_labels:
            n_detect_best,n_cond = find_opt(part_func,n_cond,False)

        n_cond_refined = n_cond
        while n_cond_refined > n_cond-1:
            n_detect_best_refined,n_cond_refined = find_opt(part_func,n_cond_refined,True)
            if n_detect_best_refined > n_labels:
                n_cond_refined = n_cond_refined + 0.1
                break


    # print(np.shape(raw_im))
    # return None
    #detect the spots
    im_mask = extrema.h_maxima(raw_im,h=n_cond_refined*h)

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
    
    print(f'Number of detected spots: {len(x)}')
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


def compute_n_param(n:float,raw_im: np.ndarray,h: float,crop_size_xy:int = 9,crop_size_z:int = 5) -> int:
    im_mask = extrema.h_maxima(raw_im,h=n*h)

    if np.shape(np.nonzero(im_mask))[1] == 0:
        print("No spots detected computed n param")
        return 0

    try:
        z,y,x = np.nonzero(im_mask)
    except ValueError as e:
        print(f'The following error has occured when trying to extract segmented pixels {e}')
        return 0
    
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

    return len(x)

def find_opt(func,n:float,flag:bool):
    if flag:
        return func(n-0.1),n-0.1
    else:
        return func(n+1),n+1


def hmax_3D_dask_h(raw_im: np.ndarray,frame: int,channel:int,labels:int,crop_size_xy:int = 4,crop_size_z:int = 4,chunk_location:tuple = (0,0,0,0,0)) -> pd.DataFrame:
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
    ims = []
    for l in np.unique(labels):
        # n_pixels.append(np.sum(labels == l))

        im_masked = raw_im * (labels == l)

        h = np.max(im_masked)

        ims.append(np.where(im_masked == h,1,0))

    im_mask = np.sum(ims,axis=0)


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
    
    print(f'Number of detected spots: {len(x)}')
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


def max5_detection(raw_im: np.ndarray,frame: int,channel:int,dilation_matrix:np.array,labels:np.array,crop_size_xy:int = 4,crop_size_z:int = 4,chunk_location:tuple = (0,0,0,0,0)) -> pd.DataFrame:
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
            im_masked_dilated = dilation_matrix * (labels == l)

            raw_im_masked_ = raw_im * (labels == l)

            raw_im_masked = raw_im_masked_*(raw_im_masked_.astype(np.uint16) == im_masked_dilated)

            new_im_flat = raw_im_masked.flatten()
            flat_index = np.argsort(new_im_flat)[-5:]

            #n_pixels=np.sum(labels == l)
            
            intensity = new_im_flat[flat_index]
            back_d = np.mean(raw_im_masked[raw_im_masked>0])
            back = np.mean(raw_im_masked_[raw_im_masked_>0])

            snr_d.extend([i/back_d for i in intensity])
            snr.append([i/back for i in intensity])

            max_coords_2.extend([np.unravel_index(f, raw_im_masked.shape) for f in flat_index])

            n_pixels.extend([np.sum(labels == l)]*5)
            labs.extend([l]*5)

    try:
        z,y,x = np.array(max_coords_2).T#np.nonzero(im_mask)
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

    # print(f'Number of detected spots: {len(x)}')
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print("No spots detected")
        return np.empty((0,17))#pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','chunk_location'])

    try:
        k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z) for i in range(len(pos))]
    except IndexError as e:
        print(f'The following error has occured when trying to create the arguments for the fitting {e}')
        return np.empty((0,17))#pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','chunk_location'])
    
    try:
        x_s,y_s,z_s= zip(*(starmap(locate_com,k)))
    except IndexError as e:
        print(f'An error occurred when trying to map the locate_com function: {e}')
        return np.empty((0,17))#pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted','frame','chunk_location'])
        
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
    df_loc['intensity'] = intensity
    df_loc['pixel_sum'] = n_pixels
    df_loc['label'] = labs
    df_loc['snr_tophat'] = snr
    df_loc['snr_dilated'] = snr_d

    return df_loc#.values