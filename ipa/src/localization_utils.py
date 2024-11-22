import scipy.optimize as opt
import numpy as np
import warnings
from skimage.transform import resize
from skimage.morphology import ball
import warnings
warnings.filterwarnings("ignore")

# Gaussian fitting functions (LSQ)

def gauss_1d(x:float, amplitude, x0, sigma, offset):
    """1D gaussian."""
    x=x
    x0 = float(x0)
    gauss = offset + amplitude * np.exp(-(x - x0) ** (2) / (2 * sigma ** (2)))
    return gauss

def gauss_2d(xy:tuple, amplitude, x0, y0, sigma_xy, offset):
    """2D gaussian."""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    gauss = offset + amplitude * np.exp(
        -(
            ((x - x0) ** (2) / (2 * sigma_xy ** (2)))
            + ((y - y0) ** (2) / (2 * sigma_xy ** (2)))
        )
    )
    return gauss

def gauss_2d_anisotropic(xy:tuple, amplitude, x0, y0, sigma_x,sigma_y, offset):
    """2D gaussian."""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    gauss = offset + amplitude * np.exp(
        -(
            ((x - x0) ** (2) / (2 * sigma_x ** (2)))
            + ((y - y0) ** (2) / (2 * sigma_y ** (2)))
        )
    )
    return gauss

def gauss_3d(xyz, amplitude, x0, y0, z0, sigma_xy, sigma_z, offset):
    """3D gaussian."""
    x, y, z = xyz
    x0 = float(x0)
    y0 = float(y0)
    z0 = float(z0)

    gauss = offset + amplitude * np.exp(
        -(
            ((x - x0) ** (2) / (2 * sigma_xy ** (2)))
            + ((y - y0) ** (2) / (2 * sigma_xy ** (2)))
            + ((z - z0) ** (2) / (2 * sigma_z ** (2)))
        )
    )
    return gauss

def find_start_end(coord, img_size, crop_size):
    '''
    Find the start and end coordinates of the crop given the spot coordinate

    Args:
    coord: coordinate of the spot
    img_size: size of the image
    crop_size: size of the crop

    Returns:
    start_dim: start coordinate of the crop
    end_dim: end coordinate of the crop

    '''
    start_dim = np.max([int(np.round(coord - crop_size // 2)), 0])
    if start_dim < img_size - crop_size:
        end_dim = start_dim + crop_size
    else:
        start_dim = img_size - crop_size
        end_dim = img_size

    return start_dim, end_dim

def get_crop(r_coord,c_coord,crop_size,image):
    '''
    Find the crop given the spot coordinates

    Args:
    r_coord: row coordinate of the spot
    c_coord: column coordinate of the spot
    crop_size: size of the crop
    image: 2D image

    Returns:
    crop: crop around the spot
    start_dim1: start coordinate of the crop in the first dimension
    end_dim1: end coordinate of the crop in the first dimension
    start_dim2: start coordinate of the crop in the second dimension
    end_dim2: end coordinate of the crop in the second dimension

    '''
    start_dim1, end_dim1 = find_start_end(c_coord, image.shape[1], crop_size)
    start_dim2, end_dim2 = find_start_end(r_coord, image.shape[0], crop_size)

    crop = image[start_dim1:end_dim1, start_dim2:end_dim2]
    
    return crop, start_dim1, end_dim1, start_dim2, end_dim2

def get_crop_3d(r_coord,c_coord,z_coord,crop_size,crop_size_z,image):
    '''
    Find the crop given the spot coordinates in 3D

    Args:
    r_coord: row coordinate of the spot
    c_coord: column coordinate of the spot
    z_coord: z coordinate of the spot
    crop_size: size of the crop
    crop_size_z: size of the crop in z
    image: 3D image

    Returns:
    crop: crop around the spot
    start_dim1: start coordinate of the crop in the first dimension
    end_dim1: end coordinate of the crop in the first dimension
    start_dim2: start coordinate of the crop in the second dimension
    end_dim2: end coordinate of the crop in the second dimension
    
    '''
    start_dim1, end_dim1 = find_start_end(c_coord, image.shape[1], crop_size)
    start_dim2, end_dim2 = find_start_end(r_coord, image.shape[2], crop_size)
    start_dim3, end_dim3 = find_start_end(z_coord, image.shape[0], crop_size_z)

    crop = image[start_dim3:end_dim3,start_dim1:end_dim1, start_dim2:end_dim2]
    
    return crop, start_dim1, end_dim1, start_dim2, end_dim2

def gauss_single_spot_1d(signal: np.ndarray) -> tuple:
    '''
    Gaussian prediction on a single crop centred on spot 1D

    Args:
    signal: 1D signal
    z_coord: z coordinate of the spot

    Returns:
    amp: amplitude of the spot
    z0: z coordinate of the spot
    sigma: standard deviation of the spot
    offset: offset of the spot
    sd: standard deviation of the z coordinate (sd of the fit)

    
    '''
    z = np.arange(0, signal.shape[0], 1)
    initial_guess = [np.max(signal),len(signal)/2,np.std(signal), np.min(signal)]
    lower = [0, 0, 0,0]
    upper = [np.max(signal)*10,len(signal),np.inf,np.max(signal)]
    bounds = [lower, upper]

    try:
        popt, pcov = opt.curve_fit(gauss_1d,z,signal,p0=initial_guess,bounds=bounds)
        sd = np.sqrt(np.diag(pcov))
    except RuntimeError:
        print('Runtime')
        return 0,0,0,0,0
    except ValueError as e:
        print('ValueError',e)
        return 0,0,0,0,0
    
    amp = popt[0]
    z0 = popt[1]
    sigma = popt[2] 
    offset = popt[3] 
    
    return amp,z0,sigma,offset,sd[1]

def gauss_single_spot_2d(crop: np.ndarray) -> tuple:
    '''
    Gaussian prediction on a single crop centred on spot 2D

    Args:
    crop: 2D crop
    c_coord: column coordinate of the spot
    r_coord: row coordinate of the spot
    crop_size: size of the crop

    Returns:
    amp: amplitude of the spot
    x0: x coordinate of the spot
    y0: y coordinate of the spot
    sigma: standard deviation of the spot
    offset: offset of the spot
    sdx: standard deviation of the x coordinate (sd of the fit)
    sdy: standard deviation of the y coordinate (sd of the fit)
    '''
    
    x = np.arange(0, crop.shape[1], 1)
    y = np.arange(0, crop.shape[0], 1)
    yy, xx = np.meshgrid(y, x,indexing="ij")

    # Guess intial parameters
    initial_guess = [np.max(crop)/2, int(crop.shape[0] // 2),int(crop.shape[1] // 2), np.std(crop), np.min(crop)*0.5]

    # Parameter search space bounds
    lower = [0, 0, 0, 0, 0]
    upper = [np.max(crop)*10,int(crop.shape[0]),int(crop.shape[1]),np.inf,np.max(crop)]
    bounds = [lower, upper]
    
    try:
        popt, pcov = opt.curve_fit(gauss_2d,(xx.ravel(), yy.ravel()),crop.ravel(),p0=initial_guess,bounds=bounds)#,max_nfev=10000)
        sd = np.sqrt(np.diag(pcov))
    
    except RuntimeError as e:
        print("Isotropic",e)
        print('Runtime')
        return 0,0,0,0,0,0,0
    except ValueError as e:
        print('ValueError',e)
        return 0,0,0,0,0,0,0
    
    amp = popt[0]
    x0 = popt[1]
    y0 = popt[2]
    sigma=popt[3]
    offset=popt[4]
    sdx = sd[1]
    sdy = sd[2]

    return amp,x0, y0, sigma,offset,sdx,sdy

def gauss_single_spot_2d_anisotropic(crop: np.ndarray) -> tuple:
    '''
    Gaussian prediction on a single crop centred on spot 2D

    Args:
    crop: 2D crop
    c_coord: column coordinate of the spot
    r_coord: row coordinate of the spot
    crop_size: size of the crop

    Returns:
    amp: amplitude of the spot
    x0: x coordinate of the spot
    y0: y coordinate of the spot
    sigma: standard deviation of the spot
    offset: offset of the spot
    sdx: standard deviation of the x coordinate (sd of the fit)
    sdy: standard deviation of the y coordinate (sd of the fit)
    '''
    
    x = np.arange(0, crop.shape[1], 1)
    y = np.arange(0, crop.shape[0], 1)
    yy, xx = np.meshgrid(y, x,indexing="ij")

    # Guess intial parameters
    initial_guess = [np.max(crop)/2, int(crop.shape[1] // 2),int(crop.shape[0] // 2), np.std(crop),np.std(crop), np.min(crop)/2]

    # Parameter search space bounds
    lower = [0, 0, 0, 0,0, 0]
    upper = [np.max(crop)*10,int(crop.shape[1]),int(crop.shape[0]),np.inf,np.inf,np.max(crop)]
    bounds = [lower, upper]
    
    try:
        popt, pcov = opt.curve_fit(gauss_2d_anisotropic,(xx.ravel(), yy.ravel()),crop.ravel(),p0=initial_guess,bounds=bounds)
        sd = np.sqrt(np.diag(pcov))
    
    except RuntimeError as e:
        print("Anisotropic",e)
        print('Runtime')
        return 0,0,0,0,0,0,0,0
    except ValueError as e:
        print('ValueError',e)
        return 0,0,0,0,0,0,0,0
    
    amp = popt[0]
    x0 = popt[1]
    y0 = popt[2]
    sigma_x=popt[3]
    sigma_y=popt[4]
    offset=popt[5]
    sdx = sd[1]
    sdy = sd[2]
    
    return amp,x0, y0, sigma_x,sigma_y,offset,sdx,sdy

def gauss_single_spot_3d(
    image: np.ndarray,
    c_coord: float,
    r_coord: float,
    z_coord: float,
    crop_size: int,
    crop_size_z: int,
    filtered_image: np.ndarray,
    raw:bool = False
) :
    '''
    Gaussian prediction on a single crop centred on spot

    Args:
    image: 3D image
    c_coord: column coordinate of the spot
    r_coord: row coordinate of the spot
    z_coord: z coordinate of the spot
    crop_size: size of the crop
    crop_size_z: size of the crop in z

    Returns:
    x0: x coordinate of the spot
    y0: y coordinate of the spot
    z0: z coordinate of the spot
    sd_x: standard deviation of the x coordinate (sd of the fit)
    sd_y: standard deviation of the y coordinate (sd of the fit)
    sd_z: standard deviation of the z coordinate (sd of the fit)

    '''

    margin=np.array((crop_size_z//2,crop_size//2,crop_size//2))
    shape=image.shape
    coords = np.array([c_coord, r_coord, z_coord])
    near_edge = np.any((coords < margin) | (coords > (shape - margin - 1)), 1)

    if near_edge:
        print('Near edge')
        return r_coord,c_coord,z_coord,0,0,0,0,0,0
    
    start_dim1, end_dim1 = find_start_end(c_coord, image.shape[1], crop_size)
    start_dim2, end_dim2 = find_start_end(r_coord, image.shape[2], crop_size)
    start_dim3, end_dim3 = find_start_end(z_coord, image.shape[0], crop_size_z)

    if raw:
        crop = image[start_dim3:end_dim3,start_dim1:end_dim1, start_dim2:end_dim2]
    else:
        crop = filtered_image[start_dim3:end_dim3,start_dim1:end_dim1, start_dim2:end_dim2]

    x = np.arange(0, crop.shape[2], 1)
    y = np.arange(0, crop.shape[1], 1)
    z = np.arange(0, crop.shape[0], 1)
    zz,yy,xx = np.meshgrid(z,y,x,indexing="ij")

    # Guess intial parameters
    x0 = int(crop.shape[2] // 2)  # Center of gaussian, middle of the crop
    y0 = int(crop.shape[1] // 2)  # Center of gaussian, middle of the crop
    z0 = int(crop.shape[0] // 2)  # Center of gaussian, middle of the crop
    
    sigma = max(*crop.shape[:-1]) * 0.1  # SD of gaussian, 10% of the crop
    sigmaz = crop.shape[-1] * 0.1  # SD of gaussian, 10% of the crop
    
    amplitude_max =np.max(crop)
    initial_guess = [amplitude_max, x0, y0, z0, sigma, sigmaz, np.min(crop)*0.5]
    # Parameter search space bounds
    lower = [0, 0, 0, 0, 0, 0,0]
    upper = [np.max(crop)*10,
        crop_size,
        crop_size,
        crop_size_z,
        np.inf,
        np.inf,
        np.max(crop),
    ]
    bounds = [lower, upper]
    try:
        popt, pcov,info_dict,mesg,ier = opt.curve_fit(
            gauss_3d,
            (xx.ravel(), yy.ravel(), zz.ravel()),
            crop.ravel(),
            p0=initial_guess,
            bounds=bounds,
            full_output=True
        )
        sd = np.sqrt(np.diag(pcov))
    except RuntimeError:
        print('Runtime')
        return r_coord,c_coord,z_coord,0,0,0,0,0,0

    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1
    z0 = popt[3] + start_dim3
    A=popt[0]
    sigma_xy=popt[4]
    sigma_z=popt[5]
    offset=popt[6]
    error=np.sum(np.abs(info_dict["fvec"])/crop.ravel())
    sd_x = sd[1]
    sd_y = sd[2]
    sd_z = sd[3]

    # If predicted spot is out of the border of the image
    if x0 >= image.shape[2] or y0 >= image.shape[1] or z0 >= image.shape[0]:
        print('Out of border')
        return r_coord,c_coord,z_coord,0,0,0,0,0,0

    return x0, y0, z0,A,sigma_xy,sigma_z,offset,error,mesg   #sd_x, sd_y, sd_z,0,0,0,0,0,0

def gauss_single_spot_2d_1d(
    image_tophat: np.ndarray,
    c_coord: float,
    r_coord: float,
    z_coord: float,
    crop_size: int,
    crop_size_z: int,
    raw_image: np.ndarray,
):
    '''
    Gaussian prediction on a single crop centred on spot 2D and 1D

    Args:
    image_tophat: 3D image
    c_coord: column coordinate of the spot
    r_coord: row coordinate of the spot
    z_coord: z coordinate of the spot
    crop_size: size of the crop
    crop_size_z: size of the crop in z
    raw_image: 3D raw image

    Returns:
    x0: x coordinate of the spot
    y0: y coordinate of the spot
    z0: z coordinate of the spot
    sx: standard deviation of the x coordinate (sd of the fit)
    sy: standard deviation of the y coordinate (sd of the fit)
    sz: standard deviation of the z coordinate (sd of the fit)
    max_spot: maximum value of the spot
    mean_back: mean value of the background
    std_back: standard deviation of the background
    max_spot_tophat: maximum value of the spot in the tophat image
    mean_back_tophat: mean value of the background in the tophat image
    std_back_tophat: standard deviation of the background in the tophat image

    This function is used to compute the sub-pixel localization of a spot in 3D using a gaussian fit in 2D for xy and 1D for z.
    It also compute the signal to noise ratio of the spot in the raw image and in the tophat image.  It returns the maximum value of the spot, the mean and standard deviation of the background in the raw image and in the tophat image. 
    '''

    start_dim1, end_dim1 = find_start_end(c_coord, image_tophat.shape[1], crop_size)
    start_dim2, end_dim2 = find_start_end(r_coord, image_tophat.shape[2], crop_size)
    start_dim3, end_dim3 = find_start_end(z_coord, image_tophat.shape[0], crop_size_z)

    signal=image_tophat[start_dim3:end_dim3,int(c_coord),int(r_coord)]
    crop = image_tophat[int(z_coord),start_dim1:end_dim1, start_dim2:end_dim2]
    
    amp_xy,x,y,sigma_xy,offset_xy,sx,sy=gauss_single_spot_2d(crop)
    amp_z,z,sigma,offset_z,sz=gauss_single_spot_1d(signal)
    
    x0 = x + start_dim2
    y0 = y + start_dim1
    z0 = z + start_dim3
    
    # If predicted spot is out of the border of the image
    if x0 >= image_tophat.shape[1] or y0 >= image_tophat.shape[2] or z0 >= image_tophat.shape[0]:
        print('Out of border')
        return 0,0,0,0,0,0,0,0,0

    # max_spot_tophat,mean_back_tophat,std_back_tophat = compute_snr(z_coord,r_coord,c_coord, image_tophat, crop_size,crop_size_z)
    # max_spot,mean_back,std_back = compute_snr(z_coord,r_coord,c_coord, raw_image, crop_size,crop_size_z)
    
    return amp_xy,x0,y0,sigma_xy,offset_xy,amp_z,z0,sigma,offset_z #sx, sy,sz,0,0,0,0,0,0

    #return x0, y0, z0, sx, sy,sy,max_spot,mean_back,std_back,max_spot_tophat,mean_back_tophat,std_back_tophat

def gauss_single_spot_2d_2d(
    image_tophat: np.ndarray,
    c_coord: float,
    r_coord: float,
    z_coord: float,
    crop_size: int,
    crop_size_z: int,
    raw_image: np.ndarray,
) :
    '''
    Gaussian prediction on a single crop centred on spot 2D and 2D

    Args:
    image_tophat: 3D image
    c_coord: column coordinate of the spot
    r_coord: row coordinate of the spot
    z_coord: z coordinate of the spot
    crop_size: size of the crop
    crop_size_z: size of the crop in z
    raw_image: 3D raw image

    Returns:
    x0: x coordinate of the spot
    y0: y coordinate of the spot
    z0: z coordinate of the spot
    sx: standard deviation of the x coordinate (sd of the fit)
    sy: standard deviation of the y coordinate (sd of the fit)
    sz: standard deviation of the z coordinate (sd of the fit)
    max_spot: maximum value of the spot
    mean_back: mean value of the background
    std_back: standard deviation of the background
    max_spot_tophat: maximum value of the spot in the tophat image
    mean_back_tophat: mean value of the background in the tophat image
    std_back_tophat: standard deviation of the background in the tophat image

    This function is used to compute the sub-pixel localization of a spot in 3D using a gaussian fit in 2D for xy and 1D for z.
    It also compute the signal to noise ratio of the spot in the raw image and in the tophat image.  It returns the maximum value of the spot, the mean and standard deviation of the background in the raw image and in the tophat image. 
    '''

    start_dim1, end_dim1 = find_start_end(c_coord, image_tophat.shape[1], crop_size)
    start_dim2, end_dim2 = find_start_end(r_coord, image_tophat.shape[2], crop_size)
    start_dim3, end_dim3 = find_start_end(z_coord, image_tophat.shape[0], crop_size_z)

    crop_yx = image_tophat[int(z_coord),start_dim1:end_dim1, start_dim2:end_dim2]
    amp_xy,x_yx,y_yx,sigma_xy,offset_xy,sx,sy=gauss_single_spot_2d(crop_yx)

    crop_zy = image_tophat[start_dim3:end_dim3,start_dim1:end_dim1,int(r_coord)]
    amp_zy,y_zy,z_zy,sigma_y_zy,sigma_z_zy,offset_zy,sz,syz=gauss_single_spot_2d_anisotropic(crop_zy)

    crop_zx = image_tophat[start_dim3:end_dim3,int(c_coord),start_dim2:end_dim2]
    amp_zx,x_zx,z_zx,sigma_x_zx,sigma_z_zx,offset_zx,sz,sxz=gauss_single_spot_2d_anisotropic(crop_zx)
    
    x0 = x_yx + start_dim2
    y0 = y_yx + start_dim1
    z0 = (z_zy+z_zx)/2 + start_dim3 
    
    # If predicted spot is out of the border of the image
    if x0 >= image_tophat.shape[1] or y0 >= image_tophat.shape[2] or z0 >= image_tophat.shape[0]:
        print('Out of border')
        return 0,0,0,0,0,0,0,0

    #max_spot_tophat,mean_back_tophat,std_back_tophat = compute_snr(z_coord,r_coord,c_coord, image_tophat, crop_size,crop_size_z)

    #max_spot,mean_back,std_back = compute_snr(z_coord,r_coord,c_coord, raw_image, crop_size,crop_size_z)
    
    return x0,y0,z0,sigma_xy,sigma_y_zy,sigma_z_zy,sigma_x_zx,sigma_z_zx #,amp_xy,sigma_xy,offset_xy,amp_zy,sigma_z_zy,sigma_y_zy,offset_zy,amp_zx,#,amp_xy,sigma_xy,offset_xy,  #amp_z,z0,sigma,offset_z #x0, y0, z0, sx, sy,sy,max_spot,mean_back,std_back,max_spot_tophat,mean_back_tophat,std_back_tophat

def compute_snr(z_coord,r_coord,c_coord, image, crop_size_xy,crop_size_z):
    '''
    Compute the signal to noise ratio of a spot

    Args:
    z_coord: z coordinate of the spot
    r_coord: row coordinate of the spot
    c_coord: column coordinate of the spot
    image: 3D image
    crop_size_xy: size of the crop in xy
    crop_size_z: size of the crop in z

    Returns:
    max_spot: maximum value of the spot
    mean_back: mean value of the background
    std_back: standard deviation of the background

    This function is used to compute the signal to noise ratio of a spot in 3D. It returns the maximum value of the spot, the mean and standard deviation of the background.
    To compute the signal to noise ratio, the function first creates a ball of radius 2 centered on the spot. 
    It then computes the maximum value of the spot and the mean and standard deviation of the background defined as the pixels not present in that ball of radius 2 centered around the spot.
    '''

    crop,*_ = get_crop_3d(r_coord,c_coord,z_coord,crop_size_xy,crop_size_z,image)

    disk_ = ball(2)

    disk_resized = resize(disk_, crop.shape, mode='constant', anti_aliasing=True)

    disk_resized = np.where(disk_resized > 0,1,0)

    crop_back = crop * (disk_resized == 0)

    crop_spot = crop * (disk_resized == 1)

    max_spot = np.max(crop_spot)
    std_back = np.std(crop_back[crop_back > 0])
    mean_back = np.mean(crop_back[crop_back > 0])

    return max_spot,mean_back,std_back

def locate_z(
    image: np.ndarray,
    y_coord: int,
    x_coord: int,
    z_coord: int,
    crop_size_z: int,
    thresh=0.6): 
    '''
    Locate the z coordinate of a spot using the center of mass 

    Args:
    image: 3D image
    y_coord: y coordinate of the spot
    x_coord: x coordinate of the spot
    z_coord: z coordinate of the spot
    crop_size_z: size of the crop in z
    thresh: threshold for the z coordinate to refine
    
    Returns:
    z: z coordinate of the spot
    '''


    
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