import logging
import os
from datetime import datetime
import numpy as np
from skimage.transform import resize
from stardist.models import StarDist2D
from csbdeep.utils import normalize


def _create_logger(path:str,name: str) -> logging.Logger:
    """
    Create logger which logs to <timestamp>-<name>.log inside the current
    working directory.

    Args: 
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Logger
    """

    if not os.path.exists(path+'/logs/'):
        os.makedirs(path+'/logs/')

    logger = logging.Logger(name.capitalize())
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    handler = logging.FileHandler(f"{path}/logs/{name}.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger



def create_logger_format(p,wildcards):
    return _create_logger(p,f"{wildcards.filename}")
    
def create_logger_workflow(wildcards):
    return _create_logger(f"{wildcards.path}",f"{wildcards.filename}_method_{wildcards.method}_cxy_{wildcards.crop_sizexy}_cz_{wildcards.crop_size_z}")

def create_logger_workflows(wildcards):
    return _create_logger(f"{wildcards.path}",f"{wildcards.filename}_cxy_{wildcards.crop_sizexy}_cz_{wildcards.crop_size_z}")

def create_logger_workflows_args(path,filename,crop_sizexy,crop_size_z):
    return _create_logger(f"{path}",f"{filename}_cxy_{crop_sizexy}_cz_{crop_size_z}")


def predict_stardist(im:np.array,size:tuple = (256,256))->np.array:
    """function to predict using stardist

    Args:
        im (np.array): the z_projected image to segment
        size (tuple): the size to expand the image (to be able to have better segmentation). Default: (256,256)

    Returns:
        np.array: the labels with the same shape as the input
    """

    #load model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Resize the image
    images_resized = np.asarray(resize(im, size, anti_aliasing=True))

    # Predict
    labels_resized, _ = model.predict_instances(normalize(images_resized))
    labels_resized = np.asarray(labels_resized)

    labels = np.asarray(resize(labels_resized, (im.shape[0], im.shape[1]), order=0))
    # # Predict
    # labels_resized, _ = zip(*[
    #         model.predict_instances(normalize(images_resized[frame, ...])) for frame in range(images_resized.shape[0])])
    # labels_resized = np.asarray(labels_resized)

    # Resize back the labels

    # labels = np.asarray([resize(labels_resized[frame, ...], (im[0].shape[1], im[0].shape[1]), order=0) for frame in range(labels_resized.shape[0])])
    return labels

def predict_stardist_complete(im:np.array,size:tuple = (976,976))->np.array:
    """function to predict using stardist

    Args:
        im (np.array): the z_projected image to segment
        size (tuple): the size to expand the image (to be able to have better segmentation). Default: (256,256)

    Returns:
        np.array: the labels with the same shape as the input
    """

    #load model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Resize the image
    if np.shape(im) != size:
        images_resized = np.asarray(
                    [resize(im[frame, ...], size, anti_aliasing=True) for frame in
                    range(im.shape[
                        0])])
    else:
        images_resized = im
    # Predict
    labels_resized, _ = zip(*[
            model.predict_instances(normalize(images_resized[frame, ...])) for frame in range(images_resized.shape[0])])
    labels_resized = np.asarray(labels_resized)

    # Resize back the labels

    labels = np.asarray([resize(labels_resized[frame, ...], (im[0].shape[1], im[0].shape[1]), order=0) for frame in range(labels_resized.shape[0])])
    return labels