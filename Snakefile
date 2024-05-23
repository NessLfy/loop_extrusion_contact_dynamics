import numpy as np
import pandas as pd
import ipa.src.preprocessing_utils
import ipa.src.correction_utils
import logging
import nd2
import glob
from datetime import datetime
import os
from ipa.src.snakemake_utils import create_logger_format,create_logger_workflow

# Define names of the input and output files
threads = config["threads"]
FILENAME = [x.split('/')[-1].split('.')[0] for x in glob.glob(f"{config['folder_path']}/*.nd2")]

images = expand("{folder_path}/{filename}.nd2", folder_path=config['folder_path'], filename="{filename}")

formatted_image = config["save_path"]+"/formatted_image_{filename}.npy"

detection = "{path}/detections_{filename}_method_{method}_cxy_{crop_sizexy}_cz_{crop_size_z}.csv"

detections_matched = expand("{path}/detections_matched_{filename}_method_{method}_cxy_{crop_sizexy}_cz_{crop_size_z}.csv", filename=FILENAME,
         method=config["method"],
         crop_sizexy=config["crop_size_xy"],
         crop_size_z= config['crop_size_z'],
         path=config["save_path"])

detection_matched = "{path}/detections_matched_{filename}_method_{method}_cxy_{crop_sizexy}_cz_{crop_size_z}.csv"

h_param = "{path}/h_params_{filename}.npy"

metadata = config['save_path']+"/{filename}_metadata.npy"

#create_l = 

rule all:
    input:
        detections_matched

rule format_images:
    input:
        images
    output:
        temp(formatted_image),
        metadata
    params:
        lo = lambda wildcards: create_logger_format(config['save_path'], wildcards)
    run:
        im = nd2.imread(input[0])
        met = nd2.ND2File(input[0])

        logger.info(f"Loaded image {input[0]}")
        params.lo.info(f"Loaded image {input[0]}")
        logger.info(f"The image has a shape of {im.shape}")
        params.lo.info(f"The image has a shape of {im.shape}")

        if("Illumination_Sequence" in input[0]):
           im=im#[1:,...]
        else:
            im=im[:,1:3,:,:]
        
        logger.info(f"Reformatted the image to have a shape of {im.shape}")
        params.lo.info(f"Reformatted the image to have a shape of {im.shape}")
        np.save(output[0], im)
        np.save(output[1],list(met.voxel_size()))

rule compute_h_param:
    input:
        formatted_image
    output:
        h_param
    params:
        lo = lambda wildcards: create_logger_format(config['save_path'], wildcards)
    run:
        h = np.zeros((2,1))
        im_big = np.load(input[0])
        logger.info(f"Loaded image {input[0]} for h param computation")
        logger.info(f"The image has a shape of {im_big.shape}")

        params.lo.info(f"Loaded image {input[0]} for h param computation")
        params.lo.info(f"The image has a shape of {im_big.shape}")

        for dim in range(np.shape(im_big)[1]):
            im = im_big[:,dim,...]
            h[dim] = ipa.src.preprocessing_utils.compute_h_param(np.expand_dims(np.max(im,axis=0),axis=0),frame=0,thresh=0.2)
        
        logger.info(f"Computed h param, the h param for channel 1 is {h[0]} and for channel 2 is {h[1]}")
        params.lo.info(f"Computed h param, the h param for channel 1 is {h[0]} and for channel 2 is {h[1]}")

        np.save(output[0],h)

rule compute_detections:
    input:
        formatted_image,
        h_param
    output:
        detection
    params:
        n = config["n"],
        thresh = config["thresh"],
        log_filename = lambda wildcards: f"{wildcards.filename}_method_{wildcards.method}_cxy_{wildcards.crop_sizexy}_cz_{wildcards.crop_size_z}",
        log_path = config["save_path"]
    threads: threads
    script:
        "ipa/src/run_detections_snakemake.py"

rule compute_matching:
    input: detection,metadata
    output: detection_matched
    params:
        cutoff = config["cutoff"],
        lo = lambda wildcards: create_logger_workflow(wildcards)
    run: 
        logger.info(f"Matching detections for {input[0]}")
        params.lo.info(f"Matching detections for {input[0]}")
        df = pd.read_csv(input[0])
        met = np.load(input[1])

        pixel_size_xy = met[0]
        pixel_size_z = met[2]
        df['x_um'] = df['x_fitted']*pixel_size_xy
        df['y_um'] = df['y_fitted']*pixel_size_xy
        df['z_um'] = df['z_fitted']*pixel_size_z

        logger.info(f'The pixel sized in xy is {pixel_size_xy} and in z is {pixel_size_z}')
        params.lo.info(f'The pixel sized in xy is {pixel_size_xy} and in z is {pixel_size_z}')

        matched = ipa.src.correction_utils.assign_closest(df[df.channel == 0],df[df.channel == 1],cutoff=params.cutoff)
        df_channel_1 = df[df.channel == 0].copy()

        logger.info(f"Matched {len(matched)} detections")
        params.lo.info(f"Matched {len(matched)} detections")

        for i in matched:
            df_channel_1.loc[i[0],'dx'] = i[2]
            df_channel_1.loc[i[0],'dy'] = i[3]
            df_channel_1.loc[i[0],'dz'] = i[4]

        df_channel_1.dropna(inplace=True,axis=0)
        df_channel_1.to_csv(output[0],index=False)