import scipy.optimize as opt
import numpy as np
from scipy.signal import convolve2d
import warnings


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
    crop_size= 4 ,
    crop_size_z= 4,
):
    """Gaussian prediction on a single crop centred on spot."""

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

EPS = 1e-4

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

def gauss_single_spot_2d(image: np.ndarray, c_coord: float, r_coord: float, crop_size=4,EPS = 1e-4) -> tuple:
    """Gaussian prediction on a single crop centred on spot."""
    start_dim1 = np.max([int(np.round(r_coord - crop_size // 2)), 0])
    if start_dim1 < len(image) - crop_size:
        end_dim1 = start_dim1 + crop_size
    else:
        start_dim1 = len(image) - crop_size
        end_dim1 = len(image)

    start_dim2 = np.max([int(np.round(c_coord - crop_size // 2)), 0])
    if start_dim2 < len(image) - crop_size:
        end_dim2 = start_dim2 + crop_size
    else:
        start_dim2 = len(image) - crop_size
        end_dim2 = len(image)

    crop = image[start_dim1:end_dim1, start_dim2:end_dim2]

    x = np.arange(0, crop.shape[1], 1)
    y = np.arange(0, crop.shape[0], 1)
    xx, yy = np.meshgrid(x, y)
  
    # Guess intial parameters
    x0 = int(crop.shape[0] // 2)  # Center of gaussian, middle of the crop 
    y0 = int(crop.shape[1] // 2)  # Center of gaussian, middle of the crop 
    sigma = max(*crop.shape) * 0.1  # SD of gaussian, 10% of the crop
    amplitude_max = max(np.max(crop) / 2, np.min(crop))  # Height of gaussian, maximum value
    initial_guess = [amplitude_max, x0, y0, sigma, 0]

    # Parameter search space bounds
    lower = [np.min(crop), 0, 0, 0, -np.inf]
    upper = [
        np.max(crop) + EPS,
        crop_size,
        crop_size,
        np.inf,
        np.inf,
    ]
    bounds = [lower, upper]
    try:
        popt, pcov = opt.curve_fit(
            gauss_2d,
            (xx.ravel(), yy.ravel()),
            crop.ravel(),
            p0=initial_guess,
            bounds=bounds,
        )
        sd = np.sqrt(np.diag(pcov))
    except RuntimeError:
        #print('Runtime')
        return r_coord, c_coord, 0,0,0
    amp = popt[0]
    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1
    sdx = sd[1]
    sdy = sd[2]

    # If predicted spot is out of the border of the image
    if x0 >= image.shape[1] or y0 >= image.shape[0]:
        return r_coord, c_coord, 0,0,0

    return y0, x0, sdx,sdy,amp

# Radial symmetry

def radialcenter(I):
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

def get_crop(r_coord,c_coord,crop_size,image):

    start_dim1, end_dim1 = find_start_end(r_coord, image.shape[0], crop_size)
    start_dim2, end_dim2 = find_start_end(c_coord, image.shape[1], crop_size)

    crop = image[start_dim1:end_dim1, start_dim2:end_dim2]
    
    return crop, start_dim1, end_dim1, start_dim2, end_dim2