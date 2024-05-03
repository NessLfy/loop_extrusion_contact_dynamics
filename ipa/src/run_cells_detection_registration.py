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


def main(path:str,name_of_run:str,thresh:float,threads:int,cutoff:float,save_path:str) -> tuple:

    logger = pre._create_logger(name='beads_detection_registration')
    logger.info(f'Starting run: {name_of_run}')
    logger.info('Starting cells detection and registration on files: {}'.format(path))

    im = nd2.imread(path)
    met = nd2.ND2File(path)

    try:
        im=im[:,:,1:3,:,:]
        logger.info('Loaded image with shape: {}'.format(im.shape))
        logger.info(f'The pixel size for this image was (x,y,z): {met.voxel_size()}')

        im_c1 = im[ :,:,0, ...]
        im_c2 = im[ :,:,1, ...]

        h = []
        h2 = []

        for frame in range(np.shape(im_c1)[0]):
            im_max_1 = np.expand_dims(np.max(im_c1[frame],axis=0),axis=0)
            im_max_2 = np.expand_dims(np.max(im_c2[frame],axis=0),axis=0)
            h.append(pre.compute_h_param(im=im_max_1,frame = 0,thresh=0.5))
            h2.append(pre.compute_h_param(im=im_max_2,frame = 0,thresh=0.5))

        # h = np.mean(h)
        # h2 = np.mean(h2)

        logger.info(f'The h parameter for channel 1 is {h} and for channel 2 is {h2}')

        
        
        detection_combined,detections_1_comb, detections_2_comb ,detections_1_fitted_comb,detections_2_fitted_comb = [],[],[],[],[]

        for frame in tqdm(range(np.shape(im_c1)[0])):
            logger.info('computing the number of detections for different thresholds')
            for n in np.arange(1,10,0.5):
                n_detections_1= len(det.hmax_3D(raw_im= im_c1,frame=frame,sd=h[frame],n = n,thresh = thresh,threads = threads,fitting=False))
                if n_detections_1 < 60:
                    n_final_1 = n
                    break
                elif n == 9.5:
                    n_final_1 = 9.5

            for n in np.arange(1,10,0.5):
                n_detections_2=len(det.hmax_3D(raw_im= im_c2,frame=frame,sd=h2[frame],n = n,thresh = thresh,threads = threads,fitting=False))
                if n_detections_2 < 60:
                    n_final_2 = n
                    break
                elif n == 9.5:
                    n_final_2 = 9.5
            logger.info(f'The n factor chosen for channel 1 is {n_final_1} and for channel 2 is {n_final_2} (minimal n so that n_detections < 60)')

            logger.info('computing the detections for channel 1')
            detections = det.hmax_3D(raw_im= im_c1,frame=frame,sd=h[frame],n = n_final_1,thresh = thresh,threads = threads,fitting=False)
            logger.info('computing the detections for channel 2')
            detections_temp = det.hmax_3D(raw_im= im_c2,frame=frame,sd=h2[frame],n = n_final_2,thresh = thresh,threads = threads,fitting=False)
            
            detections['channel'] = 1
            detections_temp['channel'] = 2

            detections_comb = pd.concat([detections, detections_temp])

            detection_combined.append(detections_comb)
            
            detections_comb['x_um'] = detections_comb['x']*met.voxel_size()[0]
            detections_comb['y_um'] = detections_comb['y']*met.voxel_size()[1]
            detections_comb['z_um'] = detections_comb['z']*met.voxel_size()[2]

            logger.info('computing the matching')
            matched = cor.assign_closest(detections_comb[detections_comb.channel ==1],detections_comb[detections_comb.channel ==2],cutoff)
            logger.info(f'{len(matched)} points were matched')

            detections_1 = detections_comb[detections_comb.channel == 1].copy()
            detections_2 = detections_comb[detections_comb.channel == 2].copy()
        

            for i in matched:
                detections_1.loc[i[0],'dx'] = i[2]
                detections_1.loc[i[0],'dy'] = i[3]
                detections_1.loc[i[0],'dz'] = i[4]
                detections_2.loc[i[1],'dx'] = i[2]
                detections_2.loc[i[1],'dy'] = i[3]
                detections_2.loc[i[1],'dz'] = i[4]


            detections_1.dropna(inplace=True, axis=0)
            detections_2.dropna(inplace=True, axis=0)

            detections_1_comb.append(detections_1)
            detections_2_comb.append(detections_2)

            detections_1_fitted = det.fitting(raw_im= im_c1,frame=frame ,detected_spot=detections_1,thresh = thresh,threads=threads)
            detections_2_fitted = det.fitting(raw_im= im_c2,frame=frame ,detected_spot=detections_2,thresh = thresh,threads=threads)

            detections_1_fitted['x_um'] = detections_1_fitted['x']*met.voxel_size()[0]
            detections_1_fitted['y_um'] = detections_1_fitted['y']*met.voxel_size()[1]
            detections_1_fitted['z_um'] = detections_1_fitted['z']*met.voxel_size()[2]

            detections_2_fitted['x_um'] = detections_2_fitted['x']*met.voxel_size()[0]
            detections_2_fitted['y_um'] = detections_2_fitted['y']*met.voxel_size()[1]
            detections_2_fitted['z_um'] = detections_2_fitted['z']*met.voxel_size()[2]

            detections_1_fitted_comb.append(detections_1_fitted)
            detections_2_fitted_comb.append(detections_2_fitted)

        detection_combined = pd.concat(detection_combined)
        detections_1_comb = pd.concat(detections_1_comb)
        detections_2_comb = pd.concat(detections_2_comb)
        detections_1_fitted_comb = pd.concat(detections_1_fitted_comb)
        detections_2_fitted_comb = pd.concat(detections_2_fitted_comb)

    except Exception as e:
        logger.error(f'Error: {e}')
        im.close()
        met.close()

    return  detection_combined,detections_1_comb, detections_2_comb ,detections_1_fitted_comb,detections_2_fitted_comb,save_path

# Execute the file
    
path = os.getcwd()

if __name__ == "__main__":

    from build_config_file import CONFIG_NAME

    with open(path+'/'+CONFIG_NAME, "r") as f:
        config = yaml.safe_load(f)

    images_path = glob.glob(f"{config['folder_path']}/*.nd2")

    config.pop('folder_path')

    for i in tqdm(images_path):
        detection_combined,detections_1_comb, detections_2_comb ,detections_1_fitted_comb,detections_2_fitted_comb,save_path = main(i,**config)
        detection_combined.to_csv(f"{save_path}/raw_detections_{i.split('/')[-1]}.csv")
        detections_1_comb.to_csv(f"{save_path}/matched_detections_channel_1_{i.split('/')[-1]}.csv")
        detections_2_comb.to_csv(f"{save_path}/matched_detections_channel_2_{i.split('/')[-1]}.csv")
        detections_1_fitted_comb.to_csv(f"{save_path}/matched_detections_channel_1_fitted_{i.split('/')[-1]}.csv")
        detections_2_fitted_comb.to_csv(f"{save_path}/matched_detections_channel_2_fitted_{i.split('/')[-1]}.csv")
