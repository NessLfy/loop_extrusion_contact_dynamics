import numpy as np
import pandas as pd
import glob
import nd2
import correction_utils as cor
import preprocessing_utils as pre
import detection_utils as det
from tqdm import tqdm
import yaml
import os


def main(path:str,name_of_run:str,n:int,thresh:float,threads:int,cutoff:float,save_path:str,method:str,crop_size_xy:int,crop_size_z:int) -> tuple:

    logger = pre._create_logger(name='beads_detection_registration')
    logger.info(f'Starting run: {name_of_run}')
    logger.info('Starting beads detection and registration on files: {}'.format(path))

    im = nd2.imread(path)
    met = nd2.ND2File(path)

    try:
        if("Illumination_Sequence" in path):
           im_c1=im[0,1:,0,...]
           im_c2=im[1,1:,1,...]
        else:
            im=im[:,1:3,:,:]
            im_c1 = im[:, 0, ...]
            im_c2 = im[:, 1, ...]
        
        logger.info('Loaded image with shape: {}'.format(im_c1.shape))
        logger.info(f'The pixel size for this image was (x,y,z): {met.voxel_size()}')

        n_detections=[]

        #logger.info('computing the number of detections for different thresholds')
        #intensity_cutoffs = np.arange(0.1,0.9,0.01)
        # logger.info(f'Intensity cutoffs: {intensity_cutoffs}')
        # for t in tqdm(intensity_cutoffs):
        #     n_detections.append(len(pre.get_loc(im=im_c1,frame = im_c1.shape[0]//2,thresh=t,mins=1.974,maxs=3)))


        h = pre.compute_h_param(im=im_c1,frame = im_c1.shape[0]//2,thresh=0.2)
        h2 = pre.compute_h_param(im=im_c2,frame = im_c2.shape[0]//2,thresh=0.2)

        logger.info(f'The h parameter for channel 1 is {h} and for channel 2 is {h2}')

        im_c1 = np.expand_dims(im_c1, axis=0)
        im_c2 = np.expand_dims(im_c2, axis=0)
        
        logger.info('computing the detections for channel 1')
        detections = det.hmax_3D(raw_im= im_c1,frame=0,sd=h,n = n,thresh = thresh,threads = threads,fitting=True,method = method,crop_size_xy=crop_size_xy,crop_size_z=crop_size_z)
        logger.info('computing the detections for channel 2')
        detections_temp = det.hmax_3D(raw_im= im_c2,frame=0,sd=h2,n = n,thresh = thresh,threads = threads,fitting=True,method = method,crop_size_xy=crop_size_xy,crop_size_z=crop_size_z)

        logger.info(f'The number of points detected in channel 1: {len(detections)}, channel 2: {len(detections_temp)}')

        detections['channel'] = 1
        detections_temp['channel'] = 2

        detections_comb = pd.concat([detections, detections_temp])
        
        detections_comb['x_um'] = detections_comb['x_fitted']*met.voxel_size()[0]
        detections_comb['y_um'] = detections_comb['y_fitted']*met.voxel_size()[1]
        detections_comb['z_um'] = detections_comb['z_fitted']*met.voxel_size()[2]

        detections_f = detections_comb

        cutoffs=np.arange(0.1,1,0.01)
        n_matched=[]

        # logger.info('computing the number of matched detections for different cutoffs')
        # logger.info(f'Cutoffs: {cutoffs}')
        # for c in tqdm(cutoffs):
        #     matched = cor.assign_closest(detections_f[detections_f.channel ==1],detections_f[detections_f.channel ==2],c)
        #     n_matched.append(len(matched))

        logger.info('computing the matching')
        matched = cor.assign_closest(detections_f[detections_f.channel ==1],detections_f[detections_f.channel ==2],cutoff)
        logger.info(f'{len(matched)} points were matched')

        detections_1 = detections_f[detections_f.channel == 1].copy()

        for i in matched:
            detections_1.loc[i[0],'dx'] = i[2]
            detections_1.loc[i[0],'dy'] = i[3]
            detections_1.loc[i[0],'dz'] = i[4]

        detections_1.dropna(inplace=True, axis=0)
        logger.info(f"Saving the detections to {save_path}")

    except Exception as e:
        logger.error(f'Error: {e}')
        im.close()
        met.close()

    return n_detections,cutoff,n_matched,detections_1

# Execute the file
    
path = os.getcwd()

if __name__ == "__main__":

    from build_config_file import CONFIG_NAME

    with open(path+'/'+CONFIG_NAME, "r") as f:
        config = yaml.safe_load(f)

    images_path = glob.glob(f"{config['folder_path']}/*.nd2")
    
    config.pop('folder_path')

    method=config['method']
    crop_size_xy=config['crop_size_xy']
    crop_size_z=config['crop_size_z']
    
    config.pop('crop_size_xy')
    config.pop('crop_size_z')

    for crop_xy,crop_z in zip(crop_size_xy,crop_size_z):
        for i in tqdm(images_path):
            n_detections,_,n_matched,detections_1 = main(path=i,crop_size_xy=crop_xy,crop_size_z=crop_z,**config)
            #np.save(f"{path}/n_detections_{i.split('/')[-1]}.npy",n_detections)
            #np.save(f"{path}/n_matched_{i.split('/')[-1]}.npy",n_matched)
            detections_1.to_csv(f"{path}/detections_{i.split('/')[-1]}_method_{method}_cxy_{crop_xy}_cz_{crop_z}.csv")

