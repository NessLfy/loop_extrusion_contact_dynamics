import pandas as pd
import numpy as np
from correction_utils import assign_closest
from pathlib import Path
import re

def find_tracks_to_refine_no_beads(track1,track2,cutoff=1.5,proportion_good_track=0.8,cutoff_jump_single_color=1):
    
    track1 = np.array(track1)[:,0:3] #Extracting the raw positions
    track2 = np.array(track2)[:,0:3] #Extracting the raw positions

    for i in range(0,len(track1)-1):
        d1=np.sqrt(np.sum((track1[i]-track1[i+1])**2))
        d2=np.sqrt(np.sum((track2[i]-track2[i+1])**2))
        if(d1>cutoff_jump_single_color or d2>cutoff_jump_single_color):
            if(i==0):
                track1[i]=[0,0,0]
                track2[i]=[0,0,0]
            else:
                if(track1[i-1][0]==0):
                    track1[i]=[0,0,0]
                    track2[i]=[0,0,0]
 
    d = (track1 - track2)
    dist_full = np.sqrt(np.sum(d**2,axis=1))
    
    if len(dist_full[(dist_full < cutoff) & (dist_full > 0)])/len(dist_full) >= proportion_good_track:
        return True
    else:
        return False
 
def correct_track_no_beads(track1,track2,df,label,snr1,snr2,pixel_size,cutoff=1.5,cutoff_jump_single_color=1):
    track1 = np.array(track1)
    track2 = np.array(track2)
    snr1 = np.array(snr1)
    snr2 = np.array(snr2)
    
    d = (track1 - track2)[:,0:3] #Extracting the raw positions
    dist_full = np.sqrt(np.sum(d**2,axis=1))
    frames_to_correct = np.where(dist_full > cutoff)[0]
    
    df[['x_um','y_um','z_um']] = df[['x','y','z']] * pixel_size
    
    pixel_size_doubled = list(pixel_size)
    pixel_size_doubled.extend(pixel_size)
    # loop over all the frames to correct (i.e. the frames where the distance between the two colors is larger than the cutoff)
    for frame in frames_to_correct:
        # get the closest points between the two colors in that frame
        m = assign_closest(df[(df.new_label == label)&(df.frame == frame)&(df.channel == 0)], df[(df.new_label == label)&(df.frame == frame)&(df.channel == 1)], cutoff)
        #we take the closest points if there is any that is below the cutoff
        if(len(m)>=1):
            d1 = df[(df.new_label == label)&(df.frame == frame)&(df.channel == 0)].reset_index(drop=True)
            d2 = df[(df.new_label == label)&(df.frame == frame)&(df.channel == 1)].reset_index(drop=True)
            small_m = np.argmin(np.sqrt(np.sum(np.array(m)[:,2:]**2,axis=1)))
            track1[frame] = d1.loc[m[small_m][0],['x','y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined']].values*pixel_size_doubled
            track2[frame] = d2.loc[m[small_m][1],['x','y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined']].values*pixel_size_doubled
            snr1[frame] = d1.loc[m[small_m][0],['snr_tophat',"sigma_xy","sigma_z"]]
            snr2[frame] = d2.loc[m[small_m][1],['snr_tophat',"sigma_xy","sigma_z"]]
        #if there is no match below the cutoff we put a gap
        else:
            track1[frame] = [0,0,0,0,0,0]
            snr1[frame] = [0,0,0]
            track2[frame] = [0,0,0,0,0,0]
            snr2[frame] = [0,0,0]

    for i in range(len(track1)): #check if the fit was not performed and put zeros if yes
        if(track1[i][2]==track1[i][-1] or track2[i][2]==track2[i][-1]):
            track1[i]=[0,0,0,0,0,0]
            track2[i]=[0,0,0,0,0,0]
            snr1[i] = [0,0,0]
            snr2[i] = [0,0,0]    
    
    for i in range(len(track1)-1): #for loop to check jumps in single color they need to be smaller than cutoff_jump_single_color
        if(track1[i][0]==0): #there is a gap -> nothing to be done
            pass
        
        elif(track1[i][0]!=0 and track1[i+1][0]==0): #frame i is not a gap but next one is
            if(i==0): #if frame i is the first one -> nothing to be done
                track1[i]=[0,0,0,0,0,0]
                track2[i]=[0,0,0,0,0,0]
                snr1[i] = [0,0,0]
                snr2[i] = [0,0,0]
            else: #check with previous frame
                d1=np.sqrt(np.sum((track1[i]-track1[i-1])[3:6]**2))
                d2=np.sqrt(np.sum((track2[i]-track2[i-1])[3:6]**2))
                if(track1[i-1][0]==0): #if previous frame is a gap we put a gap on frame i as well
                    track1[i]=[0,0,0,0,0,0]
                    track2[i]=[0,0,0,0,0,0]
                    snr1[i] = [0,0,0]
                    snr2[i] = [0,0,0]
                elif(d1>cutoff_jump_single_color or d2>cutoff_jump_single_color): #we put a gap if in one of the two colors there is a big jump
                    track1[i]=[0,0,0,0,0,0]
                    track2[i]=[0,0,0,0,0,0]
                    snr1[i] = [0,0,0]
                    snr2[i] = [0,0,0]
        else: #if the next frame is not a gap we check the jump with the next frame
            d1=np.sqrt(np.sum((track1[i]-track1[i+1])[3:6]**2))
            d2=np.sqrt(np.sum((track2[i]-track2[i+1])[3:6]**2))
            if(d1>cutoff_jump_single_color or d2>cutoff_jump_single_color): #if the jump is to big
                if(i==0): #if frame i is 0 -> we put a gap
                    track1[i]=[0,0,0,0,0,0]
                    track2[i]=[0,0,0,0,0,0]
                    snr1[i] = [0,0,0]
                    snr2[i] = [0,0,0]
                else: #otherwise we check with previous frame
                    d1=np.sqrt(np.sum((track1[i]-track1[i-1])[3:6]**2))
                    d2=np.sqrt(np.sum((track2[i]-track2[i-1])[3:6]**2))
                    if(track1[i-1][0]==0): #if previous frame is a gap we put a gap in frame i
                        track1[i]=[0,0,0,0,0,0]
                        track2[i]=[0,0,0,0,0,0]
                        snr1[i] = [0,0,0]
                        snr2[i] = [0,0,0]
                    elif(d1>cutoff_jump_single_color or d2>cutoff_jump_single_color): #we put a gap in frame i if the jump is too big
                        track1[i]=[0,0,0,0,0,0]
                        track2[i]=[0,0,0,0,0,0]
                        snr1[i] = [0,0,0]
                        snr2[i] = [0,0,0]
                    elif(d1<cutoff_jump_single_color and d2<cutoff_jump_single_color): #if in both colors the jump is smaller than the cutoff it means that there is a problem with the next one -> we put a gap on the frame i+1
                        track1[i+1]=[0,0,0,0,0,0]
                        track2[i+1]=[0,0,0,0,0,0]
                        snr1[i+1] = [0,0,0]
                        snr2[i+1] = [0,0,0]
    distances = np.array(track1 - track2)[:,3:6]
    
    return track1[:,:], track2[:,:], distances,snr1,snr2
 
def process_df_no_beads(path_run_folder,cutoff=1.5,proportion_good_track=1.0,cxy=9,cz=7,pixel_sizeinit=[0.13,0.13,0.3],raw=False,method='gauss',cutoff_single_channel=1,frame_to_cut=10):
    path_run_folder = Path(path_run_folder)
    try:
        df = pd.read_parquet(path_run_folder)
        N_frame = np.max(df.frame.unique())
        # Get the first CSV file path
        # Construct the path for the labels
        path_labels = path_run_folder.parent.with_name('label_image_tracked') / path_run_folder.name.replace('detections','label_image_tracked').replace(f'_cxy_{cxy}_cz_{cz}_method_{method}_fit_{raw}_image', '')
 
        df_labels = pd.read_parquet(path_labels)
 
        df_labels[["centroid-0","centroid-1"]]=df_labels[["centroid-0","centroid-1"]]*pixel_sizeinit[0:2]
        merged_df = pd.merge(df, df_labels[[ 'frame', 'label', 'new_label']], on=[ 'frame', 'label'], how='left')
 
        df['new_label'] = merged_df['new_label']
        
        df=df[df.frame>=frame_to_cut]
        df["frame"]=df.frame-frame_to_cut

        d = df.groupby(["new_label","channel","frame"]).apply(lambda x: x.loc[x['snr_tophat'].idxmax()])

        if method == 'com' or method =='com3d':
            d['sigma_xy'] = 0
            d['sigma_z'] = 0
 
        trajs_1 = []
        trajs_2 = []
        distances = []
        cells = []
        snr_c1 = []
        snr_c2 = []
        N_pixel = []
        label=[]
 
        pixel_size = list(pixel_sizeinit)
        pixel_size.extend(pixel_sizeinit)
 
        for track in df.new_label.unique():
            temp_traj_1 = np.zeros((N_frame+1-frame_to_cut, 6))
            temp_traj_2 = np.zeros((N_frame+1-frame_to_cut, 6))
            snr_c1_temp = np.zeros((N_frame+1-frame_to_cut,3))
            snr_c2_temp = np.zeros((N_frame+1-frame_to_cut,3))
            N_pixel_temp = np.zeros((N_frame+1-frame_to_cut,1))
            labels_to_save = np.zeros((N_frame+1-frame_to_cut,3))
            temp_traj_1[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['x', 'y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined']].values*pixel_size
            snr_c1_temp[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['snr_tophat',"sigma_xy","sigma_z"]]
            temp_traj_2[d[(d.new_label == track)&(d.channel == 1)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 1)][['x', 'y','z','x_fitted_refined','y_fitted_refined','z_fitted_refined']].values*pixel_size
            snr_c2_temp[d[(d.new_label == track)&(d.channel == 1)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 1)][['snr_tophat',"sigma_xy","sigma_z"]]

            N_pixel_temp[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['pixel_sum']].values
 
            labels_to_save[d[(d.new_label == track)&(d.channel == 0)]['frame'].values.astype(int)] = d[(d.new_label == track)&(d.channel == 0)][['label','new_label','frame']].values
            
            if find_tracks_to_refine_no_beads(temp_traj_1,temp_traj_2,cutoff,proportion_good_track,cutoff_single_channel):
                track1, track2,dist,snr1,snr2 = correct_track_no_beads(temp_traj_1,temp_traj_2,df,track,snr_c1_temp,snr_c2_temp,pixel_size[0:3],cutoff,cutoff_single_channel)
                trajs_1.append(track1)
                trajs_2.append(track2)
                distances.append(dist)
                cells.append(track)
                snr_c1.append(snr1)
                snr_c2.append(snr2)
                N_pixel.append(N_pixel_temp)
                label.append(labels_to_save)
    except FileNotFoundError:
        return [],[],[],[],[],[],[],path_run_folder.stem.replace('detections_','').replace(f'_cxy_{cxy}_cz_{cz}', '')
 
    return distances,trajs_1, trajs_2,label,snr_c1,snr_c2,N_pixel,path_run_folder.stem.replace('detections_','').replace(f'_cxy_{cxy}_cz_{cz}', '')