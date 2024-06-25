import argparse
import numpy as np
import pandas as pd
from dask.distributed import Client,LocalCluster
import dask.bag as db
import logging
from datetime import datetime
import ipa.src.snakemake_utils as snk
import ipa.src.detection_utils as det
import dask.array as da
import dask.dataframe as dd
import os
import sys
from dask import delayed
import dask
import nd2

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run detections')

    # Add the arguments
    parser.add_argument('--input_image', type=str, help='The input image file')
    parser.add_argument('--h_file', type=str, help='The h file')
    parser.add_argument('--n_file', type=str, help='The n file')
    parser.add_argument('--output_file', type=str, help='The output file')
    parser.add_argument('--log_filename', type=str, help='The log filename')
    parser.add_argument('--log_path', type=str, help='The log path')
    parser.add_argument('--crop_sizexy', type=int, help='The crop size in xy')
    parser.add_argument('--crop_sizez', type=int, help='The crop size in z')
    parser.add_argument('--threads', type=int, help='The number of threads')

    # Parse the arguments
    args = parser.parse_args()

    # Load the data
    im_big = np.load(args.input_image)
    chunk_size = (10, 15, 2, 976, 976)
    im_big_dask = da.from_array(im_big, chunks=chunk_size)

    h_ = np.load(args.h_file)
    n_ = np.load(args.n_file)

    log = snk.create_logger_workflows_args(args.log_path,args.log_filename,args.crop_sizexy,args.crop_sizez)


    log.info('Starting detection')
    log.info(f'Input image shape: {np.shape(im_big)}')
    log.info(f'Input h shape: {np.shape(h_)}')
    log.info(f'Number of threads: {args.threads}')
    log.info(f"The two n's are: {n_}")

    log.info(f'Filtered image shape: {np.shape(im_big)}')
    log.info(f'Crop size xy: {args.crop_sizexy}')
    log.info(f'Crop size z: {args.crop_sizez}')


    client = Client(n_workers=args.threads, threads_per_worker=1, processes=False)

    print(client.dashboard_link)

    try:
        def pad_output(output, max_shape=10000):
            # If the first dimension of output is smaller than max_shape
            if output.shape[0] < max_shape:
                # Calculate the padding for the first dimension
                padding = [(max_shape - output.shape[0], 0)]
                # No padding for the other dimensions
                padding.extend([(0, 0) for _ in range(len(output.shape) - 1)])
                # Pad the output array
                output_padded = np.pad(output, padding)
                print(f'shape before padding {output.shape}')
                print(f'Padded output shape: {output_padded.shape}')
                output_padded = output_padded[None,None,None,...]
                return output_padded
            else:
                print(f'shape before padding {output.shape}')
                return output
        
        def process_chunk(chunk, block_info=None):
            if block_info is not None:
                print(f'Processing chunk of shape {np.shape(chunk)} at location {block_info[0]["chunk-location"]}')
                chunk_location = block_info[0]["chunk-location"]
                result = np.concatenate([det.hmax_3D_dask(raw_im = chunk[i,:,color,...], frame = i, channel= color, sd = h_[color], n = n_[color], crop_size_xy= args.crop_sizexy, crop_size_z= args.crop_sizez,chunk_location = chunk_location) for i in range(chunk.shape[0]) for color in range(chunk.shape[2])])

                print(f'Chunk processed with shape {np.shape(result)}')
                #return result
                return pad_output(result)
            else:
                return None

        result_dask = im_big_dask.map_blocks(process_chunk, dtype=object,block_info=True)
        result = result_dask.compute()


    #client.shutdown()
    except Exception as e:
        log.error(f'The following error occured {e}')
        print(e)
        sys.exit(1)
    finally:
        client.shutdown()

    log.info('Detection finished')
    log.info(f'Output shape: {np.shape(result)}')


    output_reshaped = result.reshape(1, 1, -1, 12)
    
    output_reshaped = output_reshaped[0,0,:,:]
    print(f'Output reshaped shape: {np.shape(output_reshaped)}')


    detections = pd.DataFrame(output_reshaped,columns = ['x','y','z','x_fitted','y_fitted','z_fitted','z_fitted_refined','x_fitted_refined','y_fitted_refined','frame','channel','chunk_location'])
    detections = detections[detections.x != 0]

    detections.to_parquet(args.output_file, index=False)

if __name__ == '__main__':
    main()

