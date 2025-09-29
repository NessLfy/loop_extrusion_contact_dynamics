import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
import sys
sys.path.append('./ipa/src/')
import correction_utils as cor
import dask.array as da

import argparse


def main():
    # Step 1: Define the argument parser
    parser = argparse.ArgumentParser(description='Track cells across frames.')
    parser.add_argument('--labels_file', type=str, required=True, help='Path to the input file containing labs data.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file containing the tracked labs data.')

    # Step 2: Parse the arguments
    args = parser.parse_args()


    dfs = []

    labels = da.from_zarr(args.labels_file,component='0/')#np.load(args.labels_file)

    labels = labels.compute()
    
    for frame in range(len(labels)):
        df = pd.DataFrame(
            regionprops_table(labels[frame], properties=["label", "centroid"])
        )
        df["frame"] = frame
        dfs.append(df)
    coordinate_df = pd.concat(dfs)

    m = []
    for frame in range(len(labels)-1):
        df1 = coordinate_df[coordinate_df.frame == frame]
        df2 = coordinate_df[coordinate_df.frame == frame+1]
        matched = cor.assign_closest_cells(df1,df2,np.inf)
        m.append(matched)

        m_list = [list(np.array(x,dtype=int)[:,1]) for x in m]

        label_traj = []
        for frame in range(len(m_list)):
            for j in range(len(m_list[frame])):
                label_cell_temp = []
                if m_list[frame][j] == 0:
                    label_traj.append((frame,j+1))
                    stop = True
                elif m_list[frame][j] == -1:
                    stop = True
                else:
                    label_cell_temp.append((frame,j+1))

                    next_ = m_list[frame][j]
                    m_list[frame][j] = -1
                    t_next = frame +1
                    label_cell_temp.append((t_next,next_))
                    stop = False

                    while (not stop and t_next < len(m_list)):
                        if m_list[t_next][next_ - 1] == 0:
                            stop = True
                            label_traj.append(label_cell_temp)
                            m_list[t_next][next_ - 1] = -1
                        else:
                            next_temp = m_list[t_next][next_ - 1] 
                            m_list[t_next][next_ - 1] = -1
                            t_next = t_next + 1
                            next_ = next_temp

                            label_cell_temp.append((t_next,next_))

                            if t_next == len(m_list):
                                label_traj.append(label_cell_temp)


    for index,track in enumerate(label_traj):
        if len(np.array(track).shape)  == 1:
            coordinate_df.loc[(coordinate_df.frame == track[0])& (coordinate_df.label == track[1]),'new_label'] = index
        else:
            for l,frame in enumerate(np.array(track)[:,0]):
                coordinate_df.loc[(coordinate_df.frame == frame)& (coordinate_df.label == track[l][1]),'new_label'] = index

    coordinate_df.to_parquet(args.output_file,index=False)

if __name__ == '__main__':
    main()