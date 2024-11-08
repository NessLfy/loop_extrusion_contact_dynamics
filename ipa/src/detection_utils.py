import numpy as np
import pandas as pd
from localization_utils import gauss_single_spot_3d,gauss_single_spot_2d_1d,locate_z,gauss_single_spot_2d_2d
from itertools import starmap
import trackpy as tp
from scipy.ndimage import maximum_filter
from skimage.morphology import disk

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

def detections_beads(raw_im: np.ndarray,crop_size_xy:int = 4,crop_size_z:int = 4,fitting:bool = True,method:str = "com",threshold_percentile:float=99.99,radius_cyl_xy:int=7,radius_cyl_z:int=11,raw:bool=False) -> pd.DataFrame:
    
    d_xy=radius_cyl_xy
    d_z=radius_cyl_z

    footprint=np.zeros((d_z,disk(d_xy//2).shape[0],disk(d_xy//2).shape[0]))
    for i in range(len(footprint)):
        footprint[i]=disk(d_xy//2)
    #print(footprint.shape)
    threshold=np.percentile(raw_im,threshold_percentile)

    im_max=raw_im*np.where(raw_im>threshold,1,0)
    im_max=maximum_filter(im_max,footprint=footprint)
    im_max=raw_im*np.where(raw_im==im_max,1,0)
    
    z,y,x = np.nonzero(im_max)
    
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
        return pd.DataFrame(columns=['x','y','z','x_fitted','y_fitted','z_fitted'])
    else:
        if fitting == True:
            if(method.lower()=="gauss"):
                k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z,raw_im) for i in range(len(pos))]
                amp_xy,x_f,y_f,sigma_xy,offset_xy,amp_z,z_f,sigma,offset_z = zip(*(starmap(gauss_single_spot_2d_1d,k)))
                df_loc = pd.DataFrame([x,y,z,x_f,y_f,z_f,amp_xy,amp_z,sigma_xy,sigma,offset_xy,offset_z]).T
                df_loc.columns=['x','y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined',"A_xy","A_z","sigma_xy","sigma_z","offset_xy","offset_z"]
            
            elif method == 'gauss2d2d':
                k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z,raw_im) for i in range(len(pos))]
                x_f,y_f,z_f,sigma_xy,sigma_y_zy,sigma_z_zy,sigma_x_zx,sigma_z_zx = zip(*(starmap(gauss_single_spot_2d_2d,k)))
                df_loc = pd.DataFrame([x,y,z,x_f,y_f,z_f,sigma_xy,sigma_y_zy,sigma_z_zy,sigma_x_zx,sigma_z_zx]).T
                df_loc.columns=['x','y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined',"sigma_xy","sigma_y_zy","sigma_z_zy","sigma_x_zx","sigma_z_zx"]
            
            elif(method.lower()=="gauss3d"):
                k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z,raw_im,raw) for i in range(len(pos))]
                x_f, y_f,z_f,amp,sigma_xy,sigma_z,offset,error,msg= zip(*(starmap(gauss_single_spot_3d,k)))
                df_loc = pd.DataFrame([x,y,z,x_f, y_f,z_f,amp,sigma_xy,sigma_z,offset,error,msg]).T
                df_loc.columns=['x','y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined',"A","sigma_xy","sigma_z","offset","error","msg"]
            
            elif(method.lower()=="com3d"):
                df_loc = pd.DataFrame([x,y,z]).T
                df_loc.columns = ['x','y','z']
                try:
                    detections_refined= tp.refine.refine_com(raw_image = raw_im, image= raw_im, radius=[crop_size_z//2,crop_size_xy//2,crop_size_xy//2], coords = df_loc, max_iterations=10,
                    engine='python', shift_thresh=0.5, characterize=False,
                    pos_columns=['z','y','x'])
                except ValueError:
                   detections_refined = df_loc
                
                df_loc['x_fitted_refined'] = detections_refined['x'].values
                df_loc['y_fitted_refined'] = detections_refined['y'].values
                df_loc['z_fitted_refined'] = detections_refined['z'].values
            
            elif(method.lower()=="com"):
                df_loc = pd.DataFrame([x,y,z]).T
                df_loc.rename(columns={0:'x',1:'y',2:'z'},inplace=True)
                fitted_2d = []
                for index,row in df_loc.iterrows():
                    df_temp = df_loc[(df_loc.x == row.x)&(df_loc.y == row.y)&(df_loc.z == row.z)]
                    detections_refined= tp.refine.refine_com(raw_image = raw_im[int(row.z),...], image= raw_im[int(row.z),...], radius= [crop_size_xy//2,crop_size_xy//2], coords = df_temp, max_iterations=1,engine='python', shift_thresh=0.6, characterize=False,pos_columns=['y','x'])
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

        else:
            df_loc = pd.DataFrame([x,y,z]).T
            df_loc.rename(columns={0:'x',1:'y',2:'z'},inplace=True)

    return df_loc

def max5_detection(raw_im: np.ndarray,filtered_image:np.ndarray,frame: int,channel:int,max_filter_image:np.array,labels:np.array,crop_size_xy:int = 4,crop_size_z:int = 4,method:str = "gauss",raw:bool = False) -> pd.DataFrame:
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
    
    elif method == 'com_3d':
        try:
            detections_refined= tp.refine.refine_com(raw_image = filtered_image, image= filtered_image, radius= [crop_size_z//2,crop_size_xy//2,crop_size_xy//2], coords = df_loc, max_iterations=10,
            engine='python', shift_thresh=0.5, characterize=False,
            pos_columns=['z','y','x'])
        except ValueError:
            detections_refined = df_loc

        df_loc['x_fitted_refined'] = detections_refined['x'].values
        df_loc['y_fitted_refined'] = detections_refined['y'].values
        df_loc['z_fitted_refined'] = detections_refined['z'].values
    
    elif method == 'gauss':
        k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z,raw_im) for i in range(len(pos))]
        amp_xy,x_f,y_f,sigma_xy,offset_xy,amp_z,z_f,sigma,offset_z = zip(*(starmap(gauss_single_spot_2d_1d,k)))
        df_loc = pd.DataFrame([x,y,z,x_f,y_f,z_f,amp_xy,amp_z,sigma_xy,sigma,offset_xy,offset_z]).T
        df_loc.columns=['x','y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined',"A_xy","A_z","sigma_xy","sigma_z","offset_xy","offset_z"]

    elif method == 'gauss2d2d':
        k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z,raw_im) for i in range(len(pos))]
        x_f,y_f,z_f,sigma_xy,sigma_y_zy,sigma_z_zy,sigma_x_zx,sigma_z_zx = zip(*(starmap(gauss_single_spot_2d_2d,k)))
        df_loc = pd.DataFrame([x,y,z,x_f,y_f,z_f,sigma_xy,sigma_y_zy,sigma_z_zy,sigma_x_zx,sigma_z_zx]).T
        df_loc.columns=['x','y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined',"sigma_xy","sigma_y_zy","sigma_z_zy","sigma_x_zx","sigma_z_zx"]
    
    elif method == 'gauss3d':
        k = [(raw_im,pos[i][1],pos[i][2],pos[i][0],crop_size_xy,crop_size_z,filtered_image,raw) for i in range(len(pos))]
        x_f, y_f,z_f,amp,sigma_xy,sigma_z,offset,error,msg= zip(*(starmap(gauss_single_spot_3d,k)))
        df_loc = pd.DataFrame([x,y,z,x_f, y_f,z_f,amp,sigma_xy,sigma_z,offset,error,msg]).T
        df_loc.columns=['x','y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined',"A","sigma_xy","sigma_z","offset","error","msg"]
    
    else:
        raise Exception("Invalid fitting method. Method available: com,com3d,gauss,gauss3d")

    df_loc['frame'] = frame
    df_loc['channel'] = channel
    df_loc['intensity'] = intensity
    df_loc['pixel_sum'] = n_pixels
    df_loc['label'] = labs
    df_loc['snr_tophat'] = snr
    df_loc['snr_original'] = snr_o

    return df_loc
