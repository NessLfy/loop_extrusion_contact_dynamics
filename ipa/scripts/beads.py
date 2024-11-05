import pickle
from multiprocessing import Pool
import nd2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import trackpy as tp
from skimage.morphology import disk, white_tophat 
import detection_utils as det
import correction_utils as cor
import argparse
from pathlib import Path


def matching_beads(args):
        
    file_name_beads,crop_size_xy,crop_size_z,method,raw = args

    im = nd2.ND2File(file_name_beads)

    voxel_size = im.voxel_size()

    im = im.to_dask()

    if(len(im.shape)==5):
        N_frame=len(im)
    else:
        N_frame=1
    detections_matched=[]
    for frame in range(N_frame):#loop over time
        if(len(im.shape)==4):
            im_c1=im[:,0,...].compute()
            im_c2=im[:,1,...].compute()
        else:
            im_c1=im[frame,:,0,...].compute()
            im_c2=im[frame,:,1,...].compute()

        if(raw==True):
            pass
        else:
            im_c1=white_tophat(tp.preprocessing.lowpass(im_c1,1),footprint=np.expand_dims(disk(2),axis=0))
            im_c2=white_tophat(tp.preprocessing.lowpass(im_c2,1),footprint=np.expand_dims(disk(2),axis=0))
        
        detections_c1 = det.detections_beads(raw_im=im_c1,crop_size_xy=crop_size_xy,crop_size_z=crop_size_z,fitting=True,method=method)

        detections_c2 = det.detections_beads(raw_im=im_c2,crop_size_xy=crop_size_xy,crop_size_z=crop_size_z,fitting=True,method=method)
        print(frame,len(detections_c1),len(detections_c2))
        if(len(detections_c1)>0 and len(detections_c2)>0):
            detections_c1[["x_um","y_um","z_um"]]=voxel_size*detections_c1[["x_fitted_refined","y_fitted_refined","z_fitted_refined"]]
            detections_c2[["x_um","y_um","z_um"]]=voxel_size*detections_c2[["x_fitted_refined","y_fitted_refined","z_fitted_refined"]]
            
            matched=cor.assign_closest(detections_c1,detections_c2,0.3)
            print(frame,len(matched))
            if(len(matched)>0):
                detections_c1 = detections_c1.copy()
                for i in matched:
                    detections_c1.loc[i[0],'dx'] = i[2]
                    detections_c1.loc[i[0],'dy'] = i[3]
                    detections_c1.loc[i[0],'dz'] = i[4]
                    detections_c1.loc[i[0],'x2'] = detections_c2.loc[i[1],"x"]
                    detections_c1.loc[i[0],'y2'] = detections_c2.loc[i[1],"y"]
                    detections_c1.loc[i[0],'z2'] = detections_c2.loc[i[1],"z"]
                    try:
                        detections_c1.loc[i[0],'sigma_c2_xy'] = detections_c2.loc[i[1],"sigma_xy"]
                        detections_c1.loc[i[0],'sigma_c2_z'] = detections_c2.loc[i[1],"sigma_z"]
                    except:
                        pass
                
                detections_c1.reset_index(inplace=True,drop=True)
                detections_c1.dropna(inplace=True, axis=0)
                detections_c1["frame"]=[frame]*len(detections_c1)
                detections_c1["filename"]=[file_name_beads]*len(detections_c1)
                detections_matched.append(detections_c1)
    
    if(len(detections_matched)>0):
        detections_matched=pd.concat(detections_matched)
        return detections_matched
    else:
        return pd.DataFrame(columns=['x','y','z','frame','z_fitted_refined','x_fitted_refined','y_fitted_refined',"x_um","y_um","z_um"])    


def main():
    parser = argparse.ArgumentParser(description='Run detections on the beads')

    # Add the arguments
    parser.add_argument('--input_dir', type=str, help='The input directory to the beads')
    parser.add_argument('--output_file', type=str, help='The output file')
    parser.add_argument('--threads', type=int, help='The number of threads')
    parser.add_argument('--crop_size_xy', type=int, help='crop_size_xy')
    parser.add_argument('--crop_size_z', type=int, help='crop_size_z')
    parser.add_argument('--method', type=str, help='method for fitting')
    parser.add_argument('--raw', type=bool, help='fit on the raw image or not')

    args = parser.parse_args()

    input_path = Path(args.input_dir)

    list_files = list(input_path.glob("*.nd2"))

    output_file = args.output_file
    threads = args.threads
    method = args.method
    raw = args.raw
    crop_size_xy = args.crop_size_xy
    crop_size_z = args.crop_size_z

    tasks = [(file,crop_size_xy,crop_size_z,method,raw) for file in list_files]

    with Pool(threads) as p:
        results = p.map(matching_beads, tasks)

    
    # combining all the csv files and computing the linear regression

    df = pd.concat(results, axis=0)

    df.reset_index(inplace=True,drop=True)  
    print(df.head())

    if(len(df)>0):
        Y=df[["dx","dy","dz"]].values
        X=df[["x_um","y_um","z_um"]].values

        model = LinearRegression()
        model.fit(X,Y)

        d_x_corrected = df['dx'].values - model.predict(X)[:,0]
        d_y_corrected = df['dy'].values - model.predict(X)[:,1]
        d_z_corrected = df['dz'].values - model.predict(X)[:,2]

        df[["d_x_corrected","d_y_corrected","d_z_corrected"]]=np.vstack((d_x_corrected,d_y_corrected,d_z_corrected)).T
        
        with open(output_file,"wb") as f:
            pickle.dump(model,f)
    else:
        with open(output_file,"wb") as f:
            pickle.dump(np.zeros(3),f)



if __name__ == '__main__':
    main()