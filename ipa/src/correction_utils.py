from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np

def assign_closest(df1,df2,cutoff):
    d = distance_matrix(df1[['x_um','y_um','z_um']].values,df2[['x_um','y_um','z_um']].values)
    matched = []

    for i in range(len(d)):
        # loop over the rows of the matrix
        index = np.argsort(d[i]) # sort the row by distance (closest first)
        sort = d[i][index] # sorted distances
        for j in range(len(sort)): # loop over distances
            if sort[j] > cutoff: # if the distance is greater than the cutoff, break the loop
                break
            else:
                d_col = d[:,index[j]] # get the column of the distance matrix (corresponding to the closest spot in the other channel)
                if np.min(d_col) < sort[j]:# no match
                    pass
                else:
                    distance_vector=df1[['x','y','z']].values[i] - df2[['x','y','z']].values[index[j]]
                    matched.append((i,index[j],distance_vector[0],distance_vector[1],distance_vector[2]))
    return matched

def calculate_rototranslation_3D(reference, moving):
    """Return translation and rotation matrices.
    Args:
        reference: coordinates of fixed set of points.
        moving: coodinates of moving set of points (to which roto translation needs to be applied).

    Return:
        R, t: rotation and translation matrix
    """

    A = np.transpose(moving)
    B = np.transpose(reference)

    num_rows, num_cols = A.shape

    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t