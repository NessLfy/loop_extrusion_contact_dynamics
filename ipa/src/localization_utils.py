import scipy.optimize as opt
import numpy as np
import warnings
from skimage.transform import resize
from skimage.morphology import ball
import warnings
warnings.filterwarnings("ignore")

# Gaussian fitting functions (LSQ)

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

EPS = 1e-4

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


def gauss_single_spot(
    image: np.ndarray,
    c_coord: float,
    r_coord: float,
    z_coord: float,
    crop_size: int,
    crop_size_z: int,
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

    start_dim1, end_dim1 = find_start_end(c_coord, image.shape[1], crop_size)
    start_dim2, end_dim2 = find_start_end(r_coord, image.shape[2], crop_size)
    start_dim3, end_dim3 = find_start_end(z_coord, image.shape[0], crop_size_z)

    crop = image[start_dim3:end_dim3,start_dim1:end_dim1, start_dim2:end_dim2]

    x = np.arange(0, crop.shape[1], 1)
    y = np.arange(0, crop.shape[2], 1)
    z = np.arange(0, crop.shape[0], 1)
    zz,xx, yy = np.meshgrid(z,x, y)

    # Guess intial parameters
    x0 = int(crop.shape[1] // 2)  # Center of gaussian, middle of the crop
    y0 = int(crop.shape[2] // 2)  # Center of gaussian, middle of the crop
    z0 = int(crop.shape[0] // 2)  # Center of gaussian, middle of the crop
    sigma = max(*crop.shape[:-1]) * 0.1  # SD of gaussian, 10% of the crop
    sigmaz = crop.shape[-1] * 0.1  # SD of gaussian, 10% of the crop
    amplitude_max = max(
        np.max(crop) / 2, np.min(crop)
    )  # Height of gaussian, maximum value
    initial_guess = [amplitude_max, x0, y0, z0, sigma, sigmaz, 0]

    # Parameter search space bounds
    lower = [np.min(crop), 0, 0, 0, 0, 0, -np.inf]
    upper = [
        np.max(crop) + EPS,
        crop_size,
        crop_size,
        crop_size_z,
        np.inf,
        np.inf,
        np.inf,
    ]
    bounds = [lower, upper]
    try:
        popt, pcov = opt.curve_fit(
            gauss_3d,
            (xx.ravel(), yy.ravel(), zz.ravel()),
            crop.ravel(),
            p0=initial_guess,
            bounds=bounds,
        )
        sd = np.sqrt(np.diag(pcov))
    except RuntimeError:
        print('Runtime')
        return r_coord, c_coord, z_coord,0,0,0

    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1
    z0 = popt[3] + start_dim3
    sd_x = sd[1]
    sd_y = sd[2]
    sd_z = sd[3]

    # If predicted spot is out of the border of the image
    if x0 >= image.shape[1] or y0 >= image.shape[2] or z0 >= image.shape[0]:
        print('Out of border')
        return r_coord, c_coord, z_coord,0,0,0

    return x0, y0, z0,sd_x, sd_y, sd_z

def locate_com(
    image: np.ndarray,
    y_coord: float,
    x_coord: float,
    z_coord: float,
    crop_size: int,
    crop_size_z: int,
):
    """
    Locate the center of mass of a crop centered on a spot

    Args:
    image: 3D image
    y_coord: y coordinate of the spot
    x_coord: x coordinate of the spot
    z_coord: z coordinate of the spot
    crop_size: size of the crop
    crop_size_z: size of the crop in z

    Returns:
    x: x coordinate of the spot
    y: y coordinate of the spot
    z: z coordinate of the spot
    """
 
    start_dim1, end_dim1 = find_start_end(y_coord, image.shape[1], crop_size)
    start_dim2, end_dim2 = find_start_end(x_coord, image.shape[2], crop_size)
    start_dim3, end_dim3 = find_start_end(z_coord, image.shape[0], crop_size_z)
 
    crop = image[start_dim3:end_dim3,start_dim1:end_dim1, start_dim2:end_dim2]
    ogrid = np.ogrid[[slice(0, i) for i in crop.shape]]  # for center of mass
    ogrid = [g.astype(float) for g in ogrid]
 
    normalizer=np.sum(crop)
    
    for dim in range(crop.ndim):
        crop * ogrid[dim]
    cm=np.array([(crop * ogrid[dim]).sum() / normalizer for dim in range(crop.ndim)])
    
    x = cm[2] + start_dim2
    y = cm[1] + start_dim1
    z = cm[0] + start_dim3
 
    return x,y,z

EPS = 1e-4
# Radial symmetry

def radialcenter(I):
    '''
    Find the center of a spot using radial symmetry 2D

    Args:
    I: 2D image

    Returns:
    xc: x coordinate of the spot
    yc: y coordinate of the spot
    sigma: standard deviation of the spot
    '''
    # Number of grid points
    Ny, Nx = I.shape
    # grid coordinates are -n:n, where Nx (or Ny) = 2*n+1
    # grid midpoint coordinates are -n+0.5:n-0.5;
    xm_onerow = np.arange(-(Nx-1)/2.0+0.5, (Nx)/2.0-0.5)
    xm = np.tile(xm_onerow, (Ny-1, 1))

    ym_onecol = np.arange(-(Ny-1)/2.0+0.5, (Ny)/2.0-0.5)
    ym = np.tile(ym_onecol, (Nx-1, 1)).T
    # Calculate derivatives along 45-degree shifted coordinates (u and v)
    dIdu = I[:-1, 1:] - I[1:, :-1]
    dIdv = I[:-1, :-1] - I[1:, 1:]

    # Smoothing
    # h = np.ones((3, 3)) / 9  # simple 3x3 averaging filter
    # fdu = convolve2d(dIdu, h, mode='same')
    # fdv = convolve2d(dIdv, h, mode='same')
    fdu = dIdu
    fdv = dIdv
    
    dImag2 = fdu**2 + fdv**2  # gradient magnitude, squared

    # Slope of the gradient
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = -(fdv + fdu) / (fdu - fdv)
    # m = -(fdv + fdu) / (fdu - fdv)

    # Replace NaNs
    m = np.nan_to_num(m, nan=0.0, posinf=10.0*max(m[np.isfinite(m)]), neginf=-10.0*max(m[np.isfinite(m)]))

    # Shorthand "b"
    b = ym - m * xm

    # Weighting
    sdI2 = np.sum(dImag2)
    xcentroid = np.sum(dImag2 * xm) / sdI2
    ycentroid = np.sum(dImag2 * ym) / sdI2
    w = dImag2 / np.sqrt((xm - xcentroid)**2 + (ym - ycentroid)**2)

    # least-squares minimization
    xc, yc = lsradialcenterfit(m, b, 1)

    # Return output relative to upper left coordinate
    xc = xc + (Nx)/2.0
    yc = yc + (Ny)/2.0

    # A rough measure of the particle width
    Isub = I - np.min(I)
    px, py = np.meshgrid(np.arange(1, Nx+1), np.arange(1, Ny+1))
    xoffset = px - xc
    yoffset = py - yc
    r2 = xoffset**2 + yoffset**2
    sigma = np.sqrt(np.sum(Isub * r2) / np.sum(Isub)) / 2

    return xc, yc, sigma

def lsradialcenterfit(m, b, w):
    # least squares solution to determine the radial symmetry center
    wm2p1 = w / (m**2 + 1)
    sw = np.sum(wm2p1)
    smmw = np.sum(m**2 * wm2p1)
    smw = np.sum(m * wm2p1)
    smbw = np.sum(m * b * wm2p1)
    sbw = np.sum(b * wm2p1)
    det = smw**2 - smmw * sw
    xc = (smbw * sw - smw * sbw) / det
    yc = (smbw * smw - smmw * sbw) / det

    return xc, yc

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
    start_dim1, end_dim1 = find_start_end(r_coord, image.shape[1], crop_size)
    start_dim2, end_dim2 = find_start_end(c_coord, image.shape[2], crop_size)
    start_dim3, end_dim3 = find_start_end(z_coord, image.shape[0], crop_size_z)

    crop = image[start_dim3:end_dim3,start_dim1:end_dim1, start_dim2:end_dim2]
    
    return crop, start_dim1, end_dim1, start_dim2, end_dim2


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

def gauss_single_spot_1d(signal: np.ndarray, z_coord: float) -> tuple:
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
    lower = [0, 0, 0, -np.inf]
    upper = [1.2*np.max(signal),len(signal),np.inf,np.min(signal)]
    bounds = [lower, upper]

    try:
        popt, pcov = opt.curve_fit(gauss_1d,z,signal,p0=initial_guess,bounds=bounds)
        sd = np.sqrt(np.diag(pcov))
    except RuntimeError:
        print('Runtime')
        return z_coord,0,0,0,0
    except ValueError as e:
        print('ValueError',e)
        return z_coord,0,0,0,0
    
    amp = popt[0]
    z0 = popt[1]
    sigma = popt[2] 
    offset = popt[3] 
    
    return amp,z0,sigma,offset,sd[1]

def gauss_single_spot_2d(crop: np.ndarray, c_coord: float, r_coord: float, crop_size=4) -> tuple:
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
    xx, yy = np.meshgrid(x, y)

    # Guess intial parameters
    initial_guess = [np.max(crop), int(crop.shape[0] // 2),int(crop.shape[0] // 2), np.std(crop), np.min(crop)]

    # Parameter search space bounds
    lower = [0, 0, 0, 0, -np.inf]
    upper = [1.2*np.max(crop),crop_size,crop_size,np.inf,np.min(crop)]
    bounds = [lower, upper]
    
    try:
        popt, pcov = opt.curve_fit(gauss_2d,(xx.ravel(), yy.ravel()),crop.ravel(),p0=initial_guess,bounds=bounds)
        sd = np.sqrt(np.diag(pcov))
    
    except RuntimeError:
        print('Runtime')
        return r_coord, c_coord, 0,0,0,0,0
    except ValueError as e:
        print('ValueError',e)
        return r_coord, c_coord, 0,0,0,0,0
    
    amp = popt[0]
    x0 = popt[1]
    y0 = popt[2]
    sigma=popt[3]
    offset=popt[4]
    sdx = sd[1]
    sdy = sd[2]

    return amp,x0, y0, sigma,offset,sdx,sdy



def gauss_single_spot_2d_1d(
    image_tophat: np.ndarray,
    c_coord: float,
    r_coord: float,
    z_coord: float,
    crop_size: int,
    crop_size_z: int,
    raw_image: np.ndarray,
) :
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

    crop = image_tophat[int(z_coord),start_dim1:end_dim1, start_dim2:end_dim2]
    
    signal=image_tophat[start_dim3:end_dim3,int(c_coord),int(r_coord)]

    amp_xy,x,y,sigma_xy,offset_xy,sx,sy=gauss_single_spot_2d(crop, c_coord, r_coord, crop_size)
    amp_z,z,sigma,offset_z,sz=gauss_single_spot_1d(signal,z_coord)
    
    x0 = x + start_dim2
    y0 = y + start_dim1
    z0 = z + start_dim3
    
    # If predicted spot is out of the border of the image
    if x0 >= image_tophat.shape[1] or y0 >= image_tophat.shape[2] or z0 >= image_tophat.shape[0]:
        print(z)
        # plt.plot(signal)
        # plt.plot(gauss_1d(np.arange(len(signal)),amp_z,z,sz,offset_z))
        # plt.show()
        print('Out of border')
        return r_coord, c_coord, z_coord,0,0,0,0,0,0,0,0,0

    max_spot_tophat,mean_back_tophat,std_back_tophat = compute_snr(z_coord,r_coord,c_coord, image_tophat, crop_size,crop_size_z)

    max_spot,mean_back,std_back = compute_snr(z_coord,r_coord,c_coord, raw_image, crop_size,crop_size_z)


    return x0, y0, z0, sx, sy,sz,max_spot,mean_back,std_back,max_spot_tophat,mean_back_tophat,std_back_tophat
# Radial symmetry
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
    start_dim1, end_dim1 = find_start_end(r_coord, image.shape[0], crop_size)
    start_dim2, end_dim2 = find_start_end(c_coord, image.shape[1], crop_size)

    crop = image[start_dim1:end_dim1, start_dim2:end_dim2]
    
    return crop, start_dim1, end_dim1, start_dim2, end_dim2

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