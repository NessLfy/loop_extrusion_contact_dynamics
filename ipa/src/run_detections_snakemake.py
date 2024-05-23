import numpy as np
import pandas as pd
import detection_utils as det
import snakemake_utils as snk
from trackpy.feature import filter_image



im_big = np.load(snakemake.input[0]) # type: ignore

h_ = np.load(snakemake.input[1]) # type: ignore
threads = snakemake.threads # type: ignore
n = snakemake.params.n # type: ignore
thresh = snakemake.params.thresh # type: ignore

log_name = snakemake.params.log_filename # type: ignore
log_path = snakemake.params.log_path # type: ignore

log = snk.create_logger_workflow(snakemake.wildcards)# type: ignore

log.info('Starting detection')
log.info(f'Input image shape: {np.shape(im_big)}')
log.info(f'Input h shape: {np.shape(h_)}')
log.info(f'Number of threads: {threads}')
log.info(f'N: {n}')
log.info(f'Threshold: {thresh}')

method = snakemake.output[0].split('_')[-5] # type: ignore

crop_size_xy = int(snakemake.output[0].split('_')[-3]) # type: ignore
crop_size_z = int(snakemake.output[0].split('_')[-1].split('.')[0]) # type: ignore


 # type: ignore
log.info(f'Filtered image shape: {np.shape(im_big)}')
log.info(f'Method: {method}')
log.info(f'Crop size xy: {crop_size_xy}')
log.info(f'Crop size z: {crop_size_z}')

detect_ = []
for dim in range(np.shape(im_big)[1]):
    h = h_[dim]
    if snakemake.params.smoothing == True: #type: ignore
        im = filter_image(im_big[:,dim,...],diameter = crop_size_xy)
    else:
        im = im_big[:,dim,...]
    im = np.expand_dims(im, axis=0)
    detections = det.hmax_3D(raw_im= im,
    frame=0,sd=h,n = n,
    thresh = thresh,threads = threads,
    fitting=True,method = method,
    crop_size_xy=crop_size_xy,
    crop_size_z=crop_size_z)
    if len(detections) == 0:
        log.warning(f'No detections found for channel {dim}')
        continue
    detections['channel'] = dim
    detect_.append(detections)
    log.info(f'Finished detection for channel {dim}, {len(detections)} detections found and fitted')


detect_ = pd.concat(detect_)
detect_.to_csv(snakemake.output[0],index=False) # type: ignore
