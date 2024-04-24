from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np

def assign_closest(df1,df2,cutoff):
    d = distance_matrix(df1[['x','y','z']].values,df2[['x','y','z']].values)
    matched = []

    for i in range(len(d)):
        # loop over the rows of the matrix
        index = np.argsort(d[i]) # sort the row by distance (closest first)
        sort = d[i][index] # sorted distances
        for j in range(len(sort)): # loop over distances
            d_col = d[:,index[j]] # get the column of the distance matrix (corresponding to the closest spot in the other channel)
            if sort[j] > cutoff: # if the distance is greater than the cutoff, break the loop
                break
            else:
                if np.min(d_col) < sort[j]:
                    # no match
                    pass
                else:
                    #matched.append([i,index[j],sort[j]])
                    matched.append([i,index[j],sort[j],df1[['x','y','z']].values[i] - df2[['x','y','z']].values[index[j]]])
                    #print(df1[['x','y','z']].values[i] - df2[['x','y','z']].values[index[j]])
    return matched
