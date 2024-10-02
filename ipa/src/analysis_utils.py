import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
from correction_utils import assign_closest
from pathlib import Path
import re


def split_using_track(vec_to_split,track,min_track_length=10):
    split_points = np.where(track[:,0] == 0)[0]

    if len(split_points) == 0:
        vec_split = [vec_to_split]
    else:
        vec_split=[]
        for i in range(len(split_points)):
            if(split_points[i]==0):
                pass
            else:
                if(i==0):
                    diff=split_points[i]-0
                    if(diff>min_track_length):
                        vec_split.append(vec_to_split[0:split_points[i]])
                else:
                    diff=split_points[i]-split_points[i-1]
                    if(diff>min_track_length):
                        vec_split.append(vec_to_split[split_points[i-1]+1:split_points[i]])
        if split_points[-1] < len(vec_to_split)-min_track_length-1:
            vec_split.append(vec_to_split[split_points[-1]+1:])
    
    return vec_split

def split_tracks(track,min_track_length=10):
    split_points = np.where(track[:,0] == 0)[0]

    if len(split_points) == 0:
        track_split = [track]
    else:
        track_split=[]
        for i in range(len(split_points)):
            if(split_points[i]==0):
                pass
            else:
                if(i==0):
                    diff=split_points[i]-0
                    if(diff>min_track_length):
                        track_split.append(track[0:split_points[i]])
                else:
                    diff=split_points[i]-split_points[i-1]
                    if(diff>min_track_length):
                        track_split.append(track[split_points[i-1]+1:split_points[i]])
        if split_points[-1] < len(track)-min_track_length-1:
            track_split.append(track[split_points[-1]+1:])
    
    return track_split

def compute_msd(trajs,files,N_dimensions=3,min_track_length=10):
    msd_WT=[[]for i in range(480)]#np.zeros(480)
    msd_dtag=[[] for i in range(480)]#np.zeros(480)
    
    for movie in range(len(trajs)):
        f=files[movie]
        for track_index in range(len(trajs[movie])):
            
            track=np.array(trajs[movie][track_index])[:,0:N_dimensions]
            track_split = split_tracks(track,min_track_length=min_track_length)
            
            for i in range(len(track_split)):
                d=distance_matrix(track_split[i],track_split[i])**2
                
                for j in range(1,len(d)):
                    delta=np.diag(d,k=j)
                    if '1A2aux' in f or 'dtag' in f:
                        msd_dtag[j].append(np.mean(delta)) #=#.append((j,np.log(np.mean(delta))))
                    else:
                        msd_WT[j].append(np.mean(delta))

    msd_WT=[np.mean(x) for x in msd_WT[1:]]
    msd_dtag=[np.mean(x) for x in msd_dtag[1:]]
    
    return np.array(msd_WT),np.array(msd_dtag)

def compute_msd_kepten(trajs,files,N_dimensions=3,min_track_length=10):
    msd_WT=[[]for i in range(480)]#np.zeros(480)
    msd_dtag=[[] for i in range(480)]#np.zeros(480)
    
    for movie in range(len(trajs)):
        f=files[movie]
        for track_index in range(len(trajs[movie])):
            
            track=np.array(trajs[movie][track_index])[:,0:N_dimensions]
            track_split = split_tracks(track,min_track_length=min_track_length)
            
            for i in range(len(track_split)):
                d=distance_matrix(track_split[i],track_split[i])**2
                
                for j in range(1,len(d)):
                    delta=np.diag(d,k=j)
                    if '1A2aux' in f or 'dtag' in f:
                        msd_dtag[j].append(np.log(np.mean(delta))) #=#.append((j,np.log(np.mean(delta))))
                    else:
                        msd_WT[j].append(np.log(np.mean(delta)))

    msd_WT=[np.exp(np.mean(x)) for x in msd_WT[1:]]
    msd_dtag=[np.exp(np.mean(x)) for x in msd_dtag[1:]]
    
    return np.array(msd_WT),np.array(msd_dtag)

def find_tracks_to_refine(track1,track2,model,cutoff=1.0,proportion_good_track=0.8):
    t_1 = np.array(track1)
    t_2 = np.array(track2)

    coord_zero_1 = np.where(t_1[:,0] == 0)[0]
    coord_zero_2 = np.where(t_2[:,0] == 0)[0]

    d = t_1 - t_2
    d[coord_zero_1] = [0,0,0]
    d[coord_zero_2] = [0,0,0]

    d_x2 = np.array(d)[:,0] - model.predict(t_1)[:,0]
    d_y2 = np.array(d)[:,1] - model.predict(t_1)[:,1]
    d_z2 = np.array(d)[:,2] - model.predict(t_1)[:,2]

    d_corrected = np.vstack((d_x2,d_y2,d_z2)).T
    d_corrected[coord_zero_1] = [0,0,0]
    d_corrected[coord_zero_2] = [0,0,0]

    dist_full = np.sqrt(np.sum((d_corrected*(0.13,0.13,0.3))**2,axis=1))

    if len(dist_full[(dist_full < cutoff) & (dist_full > 0)])/len(dist_full) >= proportion_good_track:
        return True
    else:
        return False

def correct_track(track1,track2,model,df,label,cutoff=1.0):
    track1 = np.array(track1)
    track2 = np.array(track2)

    coord_zero_1 = np.where(track1[:,0] == 0)[0]
    coord_zero_2 = np.where(track2[:,0] == 0)[0]

    d = track1 - track2
    d[coord_zero_1] = [0,0,0]
    d[coord_zero_2] = [0,0,0]

    d_x2 = np.array(d)[:,0] - model.predict(track1)[:,0]
    d_y2 = np.array(d)[:,1] - model.predict(track1)[:,1]
    d_z2 = np.array(d)[:,2] - model.predict(track1)[:,2]

    d_corrected = np.vstack((d_x2,d_y2,d_z2)).T
    d_corrected[coord_zero_1] = [0,0,0]
    d_corrected[coord_zero_2] = [0,0,0]

    dist_full = np.sqrt(np.sum((d_corrected*(0.13,0.13,0.3))**2,axis=1))

    frames_to_correct = np.where(dist_full > cutoff)[0]

    df[['x_um','y_um','z_um']] = df[['x_fitted_refined','y_fitted_refined','z_fitted_refined']] * [0.13,0.13,0.3]

    for frame in frames_to_correct:
        
        m = assign_closest(df[(df.new_label == label)&(df.frame == frame)&(df.channel == 0)], df[(df.new_label == label)&(df.frame == frame)&(df.channel == 1)], cutoff)
        
        if frame == 0:
            if(len(m)>=1):
                d1 = df[(df.new_label == label)&(df.frame == frame)&(df.channel == 0)].reset_index(drop=True)
                d2 = df[(df.new_label == label)&(df.frame == frame)&(df.channel == 1)].reset_index(drop=True)
                small_m = np.argmin(np.sqrt(np.sum(np.array(m)[:,2:]**2,axis=1)))
                track1[frame] = d1.loc[m[small_m][0],['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values
                track2[frame] = d2.loc[m[small_m][1],['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values
            else:
                track1[frame] = [0,0,0]
                track2[frame] = [0,0,0]
        
        elif(len(m) > 1):
            
            df_temp_1 = pd.DataFrame(track1[frame - 1]*[0.13,0.13,0.3]).T
            df_temp_1.columns=['x_um','y_um','z_um']
            df_temp_2 = pd.DataFrame(track2[frame - 1]*[0.13,0.13,0.3]).T
            df_temp_2.columns=['x_um','y_um','z_um']
            
            d1 = df[(df.new_label == label)&(df.frame == frame)&(df.channel == 0)].reset_index(drop=True)
            d2 = df[(df.new_label == label)&(df.frame == frame)&(df.channel == 1)].reset_index(drop=True)
        
            if (df_temp_1['x_um'].values[0] == 0) or (df_temp_2['x_um'].values[0] == 0):
                small_m = np.argmin(np.sqrt(np.sum(np.array(m)[:,2:]**2,axis=1)))
                track1[frame] = d1.loc[m[small_m][0],['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values
                track2[frame] = d2.loc[m[small_m][1],['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values
            else:
                d1 = d1.loc[np.array(m)[:,0],:].reset_index(drop=True)
                d2 = d2.loc[np.array(m)[:,1],:].reset_index(drop=True)
                m1 =  assign_closest(d1, df_temp_1, 1.0)
                m2 =  assign_closest(d2, df_temp_2, 1.0)
            
                if len(m1) == 0:
                    track1[frame] = [0,0,0]
                    track2[frame] = [0,0,0]
                elif len(m2) == 0:
                    track1[frame] = [0,0,0]
                    track2[frame] = [0,0,0]
                else:
                    track1[frame] = d1.loc[m1[0][0],['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values
                    track2[frame] = d2.loc[m2[0][0],['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values
        
        elif len(m) == 1:
            df_temp_1 = pd.DataFrame(track1[frame - 1]*[0.13,0.13,0.3]).T
            df_temp_1.columns=['x_um','y_um','z_um']
            df_temp_2 = pd.DataFrame(track2[frame - 1]*[0.13,0.13,0.3]).T
            df_temp_2.columns=['x_um','y_um','z_um']
            
            d1 = df[(df.new_label == label)&(df.frame == frame)&(df.channel == 0)].reset_index(drop=True)
            d1 = pd.DataFrame(d1.loc[m[0][0],['x_fitted_refined','y_fitted_refined','z_fitted_refined']])
            d1=d1.T
            d1.reset_index(drop=True,inplace=True)
            d1.columns=['x_fitted_refined','y_fitted_refined','z_fitted_refined']
            d1[["x_um","y_um","z_um"]]=d1[['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values*[0.13,0.13,0.3]
            
            d2 = df[(df.new_label == label)&(df.frame == frame)&(df.channel == 1)].reset_index(drop=True)
            d2 = pd.DataFrame(d2.loc[m[0][1],['x_fitted_refined','y_fitted_refined','z_fitted_refined']])
            d2=d2.T
            d2.reset_index(drop=True,inplace=True)
            d2.columns=['x_fitted_refined','y_fitted_refined','z_fitted_refined']
            d2[["x_um","y_um","z_um"]]=d2[['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values*[0.13,0.13,0.3]
            
            if (df_temp_1['x_um'].values[0] == 0) or (df_temp_2['x_um'].values[0] == 0):
                track1[frame] = d1.loc[0,['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values
                track2[frame] = d2.loc[0,['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values
            
            else:
                m1 =  assign_closest(d1, df_temp_1, 1.0)
                m2 =  assign_closest(d2, df_temp_2, 1.0)
                if len(m1) == 0:
                    track1[frame] = [0,0,0]
                    track2[frame] = [0,0,0]
                elif len(m2) == 0:
                    track1[frame] = [0,0,0]
                    track2[frame] = [0,0,0]
                else:
                    track1[frame] = d1.loc[m1[0][0],['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values
                    track2[frame] = d2.loc[m2[0][0],['x_fitted_refined','y_fitted_refined','z_fitted_refined']].values        
        
        else:
            track1[frame] = [0,0,0]
            track2[frame] = [0,0,0]
    
    coord_zero_1 = np.where(track1[:,0] == 0)[0]
    coord_zero_2 = np.where(track2[:,0] == 0)[0]

    d = track1 - track2
    d[coord_zero_1] = [0,0,0]
    d[coord_zero_2] = [0,0,0]

    d_x2 = np.array(d)[:,0] - model.predict(track1)[:,0]
    d_y2 = np.array(d)[:,1] - model.predict(track1)[:,1]
    d_z2 = np.array(d)[:,2] - model.predict(track1)[:,2]

    d_corrected = np.vstack((d_x2,d_y2,d_z2)).T
    d_corrected[coord_zero_1] = [0,0,0]
    d_corrected[coord_zero_2] = [0,0,0]

    dist_corrected = d_corrected*(0.13,0.13,0.3)
    
    #dist_corrected[dist_corrected > cutoff] = 0
    
    return track1, track2, dist_corrected

def process_df(path_run_folder,cutoff=0.3,proportion_good_track=1.0):
    path_run_folder = Path(path_run_folder)
    df = pd.read_parquet(path_run_folder)
    N_frame = np.max(df.frame.unique())
    # Get the first CSV file path
    # Construct the path for the labels
    path_labels = path_run_folder.parent.with_name('label_image_tracked') / path_run_folder.name.replace('detections','label_image_tracked').replace('_cxy_9_cz_7', '')

    # Construct the path for the beads

    # Extract the stem
    stem = path_run_folder.stem

    # Define the date pattern (e.g., YYYYMMDD)
    date_pattern = r'\d{8}'

    # Search for the date in the stem
    match = re.search(date_pattern, stem)

    path_beads = path_run_folder.parent.with_name('beads') / ('3d_linear_regression_' + match.group() + '.pkl')

    df_labels = pd.read_parquet(path_labels)
    df_labels[["centroid-0","centroid-1"]]=df_labels[["centroid-0","centroid-1"]]*(0.13,0.13)
    merged_df = pd.merge(df, df_labels[[ 'frame', 'label', 'new_label']], on=[ 'frame', 'label'], how='left')

    df['new_label'] = merged_df['new_label']
    d = df.groupby(["new_label","channel","frame"]).apply(lambda x: x.loc[x['snr_tophat'].idxmax()])

    with open(path_beads,'rb') as r:
        model = pickle.load(r)

    trajs_1 = []
    trajs_2 = []
    distances = []
    cells = []
    snr_c1 = []
    snr_c2 = []
    N_pixel = []

    for track in d.new_label.unique():
        temp_traj_1 = np.zeros((N_frame+1, 3))
        temp_traj_2 = np.zeros((N_frame+1, 3))
        snr_c1_temp = np.zeros((N_frame+1,1))
        snr_c2_temp = np.zeros((N_frame+1,1))
        N_pixel_temp = np.zeros((N_frame+1,1))
        
        temp_traj_1[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['x_fitted_refined', 'y_fitted_refined','z_fitted_refined']].values
        snr_c1_temp[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['snr_tophat']].values

        temp_traj_2[d[(d.new_label == track)&(d.channel == 1)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 1)][['x_fitted_refined', 'y_fitted_refined','z_fitted_refined']].values
        snr_c2_temp[d[(d.new_label == track)&(d.channel == 1)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 1)][['snr_tophat']].values

        N_pixel_temp[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['pixel_sum']].values

        if find_tracks_to_refine(temp_traj_1,temp_traj_2,model,cutoff,proportion_good_track):
            track1, track2,dist = correct_track(temp_traj_1,temp_traj_2,model,df,track,cutoff)
            trajs_1.append(track1*(0.13,0.13,0.3))
            trajs_2.append(track2*(0.13,0.13,0.3))
            distances.append(dist)
            cells.append(track)
            snr_c1.append(snr_c1_temp)
            snr_c2.append(snr_c2_temp)
            N_pixel.append(N_pixel_temp)

    return distances,trajs_1, trajs_2,df_labels[df_labels.new_label.isin(cells)],snr_c1,snr_c2,N_pixel,path_run_folder.stem.replace('detections_','').replace('_cxy_9_cz_7', '') 

def process_df_1b2(path_run_folder,cutoff=0.3,proportion_good_track=0.8):
    path_run_folder = Path(path_run_folder)
    df = pd.read_parquet(path_run_folder)
    N_frame = np.max(df.frame.unique())
    # Get the first CSV file path
    # Construct the path for the labels
    path_labels = path_run_folder.parent.with_name('label_image_tracked') / path_run_folder.name.replace('detections','label_image_tracked').replace('_cxy_9_cz_7', '')
    # Construct the path for the beads

    # Extract the stem
    stem = path_run_folder.stem

    # Define the date pattern (e.g., YYYYMMDD)
    date_pattern = r'\d{8}'

    # Search for the date in the stem
    match = re.search(date_pattern, stem)

    path_beads = path_run_folder.parent.with_name('beads') / ('3d_linear_regression_' + match.group() + '.pkl')
    df_labels = pd.read_parquet(path_labels)
    df_labels[["centroid-0","centroid-1"]]=df_labels[["centroid-0","centroid-1"]]*(0.13,0.13)
    merged_df = pd.merge(df, df_labels[[ 'frame', 'label', 'new_label']], on=[ 'frame', 'label'], how='left')

    df['new_label'] = merged_df['new_label']
    d = df.groupby(["new_label","channel","frame"]).apply(lambda x: x.loc[x['snr_tophat'].idxmax()])

    with open(path_beads,'rb') as r:
        model = pickle.load(r)

    trajs_1 = []
    trajs_2 = []
    distances = []
    cells = []
    snr_c1 = []
    snr_c2 = []
    N_pixel = []
    for track in d.new_label.unique():
        temp_traj_1 = np.zeros((N_frame+1, 3))
        temp_traj_2 = np.zeros((N_frame+1, 3))
        snr_c1_temp = np.zeros((N_frame+1,4))
        snr_c2_temp = np.zeros((N_frame+1,4))
        N_pixel_temp = np.zeros((N_frame+1,1))
        
        temp_traj_1[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['x_fitted_refined', 'y_fitted_refined','z_fitted_refined']].values*(0.13,0.13,0.3)
        snr_c1_temp[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['snr_tophat','max_original','mean_back_original','std_back_original']].values

        temp_traj_2[d[(d.new_label == track)&(d.channel == 1)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 1)][['x_fitted_refined', 'y_fitted_refined','z_fitted_refined']].values*(0.13,0.13,0.3)
        snr_c2_temp[d[(d.new_label == track)&(d.channel == 1)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 1)][['snr_tophat','max_original','mean_back_original','std_back_original']].values

        N_pixel_temp[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['pixel_sum']].values

        if find_tracks_to_refine(temp_traj_1,temp_traj_2,model,cutoff,proportion_good_track):
            track1, track2,dist = correct_track(temp_traj_1,temp_traj_2,model,df,track,cutoff)

            temp_traj_1[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['x', 'y','z']].values
            temp_traj_2[d[(d.new_label == track)&(d.channel == 1)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 1)][['x', 'y','z']].values
            trajs_1.append(track1)
            trajs_2.append(track2)
            distances.append(dist)
            cells.append(track)
            snr_c1.append(snr_c1_temp)
            snr_c2.append(snr_c2_temp)
            N_pixel.append(N_pixel_temp)

    #return distances,trajs_1, trajs_2,df_labels[df_labels.new_label.isin(cells)],snr_c1,snr_c2,N_pixel,path_run_folder.stem.replace('detections_','').replace('_cxy_9_cz_7', '') 
    return distances,trajs_1, trajs_2,df_labels[df_labels.new_label.isin(cells)],snr_c1,snr_c2,N_pixel,path_run_folder.stem.replace('detections_','').replace('_cxy_9_cz_7', '') 