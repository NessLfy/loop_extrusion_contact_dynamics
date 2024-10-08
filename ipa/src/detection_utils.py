import numpy as np
import pandas as pd
from multiprocessing import Pool
from skimage.morphology import extrema
from functools import partial
from localization_utils import gauss_single_spot,gauss_single_spot_2d,locate_com,gauss_single_spot_2d_1d,locate_z
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


def process_labels(l,labels,raw_im,max_filter_image,filtered_image):
    '''
    Function to process each individual label

    Args:
    l: int, the label to process
    labels: np.array, the labels of the image
    raw_im: np.array, the raw image
    max_filter_image: np.array, the maximum filter image
    filtered_image: np.array, the filtered image (tophat + lowpass)

    Returns:
    max_coords_2: list, the coordinates of the 5 brightest pixels
    n_pixels: list, the number of pixels in the label
    snr: list, the signal to noise ratio of the 5 brightest pixels in the filtered image (tophat + lowpass)
    snr_o: list, the signal to noise ratio of the 5 brightest pixels in the raw image 

    This function processes each label in the image and returns the coordinates of the 5 brightest pixels, the number of pixels in the label,
    the signal to noise ratio of the 5 brightest pixels in the filtered image and the signal to noise ratio of the 5 brightest pixels in the raw image
    '''

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

def max5_detection(raw_im: np.ndarray,filtered_image:np.ndarray,frame: int,channel:int,max_filter_image:np.array,labels:np.array,crop_size_xy:int = 4,crop_size_z:int = 4,method:str = "gauss") -> pd.DataFrame:
    """
    Function to detect the 5 brightest pixels in each label and make a dataframe of the sub-pixel localizations

    Args:
        raw_im (np.array): the raw image to detect spots in
        filtered_image (np.array): the filtered image to detect spots in
        frame (int): the frame to segment
        channel (int): the channel to segment
        max_filter_image (np.array): the maximum filter image
        labels (np.array): the labels of the image
        crop_size_xy (int): the size of the crop in the xy plane
        crop_size_z (int): the size of the crop in the z plane
        method (str): the method to use for fitting the spots    
        
    Returns:
        pd.DataFrame: Dataframe of sub-pixel localizations of the detected spots

    This function detects the 5 brightest pixels in each label and makes a dataframe of the sub-pixel localizations. The function can use either the center of mass or a gaussian fit to fit the spots.
    The function loops over all labels in the image and process them using the function process_labels. 
    The function then checks if the detected spots are close to the edge of the image and removes them if they are.
    The function then fits the spots using either the center of mass or a gaussian fit and returns a dataframe of the sub-pixel localizations.
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
