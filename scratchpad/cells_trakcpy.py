import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../ipa/src/')
import preprocessing_utils as pre
import localization_utils as loc
import detection_utils as det
import nd2
import correction_utils as cor
import stackview
from glob import glob
from tqdm import tqdm
import seaborn as sns
from sklearn import linear_model
from scipy.spatial import distance_matrix
from skimage import  restoration
import trackpy as tp
import pims
from skimage import data, restoration, util
from skimage.filters import gaussian

path_dir="/tungstenfs/scratch/ggiorget/kristina/Microscopy/"
dates=["20240506"]
bead_diameter=[5,7,9,11,13]

detections=[[[]for j in range(len(bead_diameter))]for i in range(len(dates))]
for date_index,date in enumerate(dates):
    path=path_dir+date
    images_path = glob(path+"/B2/*.nd2")
    for diameter_index,diameter in enumerate(bead_diameter):
        detection_temp=[]
        for image in images_path:
            im=nd2.imread(image)
            if("Illumination_Sequence" in image):
                im_c1=im[0,:,0,...]
                im_c2=im[0,:,1,...]
            else:
                if(len(im.shape)>4):
                    im=im[0]
                im=im[:,1:3,:,:]
                im_c1 = im[:, 0, ...]
                im_c2 = im[:, 1, ...]
            met = nd2.ND2File(image)
            min_start=100
            feature_1=tp.locate(im_c1,diameter=diameter,minmass=min_start,preprocess=True)
            if(len(feature_1)==0):
                pass
            else:
                while(len(feature_1)>20):
                    min_start+=1000
                    feature_1=tp.locate(im_c1,diameter=diameter,minmass=min_start,preprocess=True)
                min_start-=1000
                feature_1=tp.locate(im_c1,diameter=diameter,minmass=min_start,preprocess=True)
                while(len(feature_1)>20):
                    min_start+=100
                    feature_1=tp.locate(im_c1,diameter=diameter,minmass=min_start,preprocess=True)
                feature_1.drop(columns=['ecc'],inplace=True)
                feature_1.reset_index(inplace=True)
            
            min_start=100
            feature_2=tp.locate(im_c2,diameter=diameter,minmass=min_start,preprocess=True)
            if(len(feature_2)==0):
                pass
            else:
                while(len(feature_2)>20):
                    min_start+=1000
                    feature_2=tp.locate(im_c2,diameter=diameter,minmass=min_start,preprocess=True)
                min_start-=1000
                feature_2=tp.locate(im_c2,diameter=diameter,minmass=min_start,preprocess=True)
                while(len(feature_2)>20):
                    min_start+=100
                    feature_2=tp.locate(im_c2,diameter=diameter,minmass=min_start,preprocess=True)
                feature_2.drop(columns=['ecc'],inplace=True)
                feature_2.reset_index(inplace=True)
            
            if(len(feature_1)>0 and len(feature_2)>0):
                feature_1['x_um'] = feature_1['x']*met.voxel_size()[0]
                feature_1['y_um'] = feature_1['y']*met.voxel_size()[1]
                feature_1['z_um'] = feature_1['z']*met.voxel_size()[2]
                feature_1["movie"]=image.split("/")[-1]
                feature_2['x_um'] = feature_2['x']*met.voxel_size()[0]
                feature_2['y_um'] = feature_2['y']*met.voxel_size()[1]
                feature_2['z_um'] = feature_2['z']*met.voxel_size()[2]

                matched=cor.assign_closest(feature_1,feature_2,0.5)
                if(len(matched)):
                    feature_1_final = feature_1.copy()

                    for i in matched:
                        feature_1_final.loc[i[0],'dx'] = i[2]
                        feature_1_final.loc[i[0],'dy'] = i[3]
                        feature_1_final.loc[i[0],'dz'] = i[4]
                        mass_2=feature_2.loc[i[1],'mass']
                        feature_1_final.loc[i[0],'mass2'] = mass_2

                    feature_1_final.dropna(inplace=True, axis=0)
                    detection_temp.append(feature_1_final)
        detections[date_index][diameter_index]=pd.concat(detection_temp)
        detections[date_index][diameter_index].to_csv("DATA_CELLS/"+date+"_"+str(diameter)+".csv")