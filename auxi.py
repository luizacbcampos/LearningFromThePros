import re
import ast
import cv2
import math
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mplsoccer import VerticalPitch
import statsmodels.api as sm

import gkpose as gk

def import_keypoints(path_2d='data/pose/pose_1v1_2d.csv', path_3d='data/pose/pose_1v1_3d.csv'):
	'''
		Imports 2D and 3D keypoints
	'''
	set_3d_df = pd.read_csv(path_3d, index_col=0)
	set_3d_df = gk.getPhotoID(set_3d_df)

	set_2d_df = pd.read_csv(path_2d, index_col=0)
	set_2d_df = gk.getPhotoID(set_2d_df)
	
	return set_2d_df, set_3d_df

def import_StatsBomb_1v1_data(path='data/events/1v1_events.csv'):
	'''
		Import StatsBomb 1v1 Data
	'''
	converter = {
	'location':ast.literal_eval,
	'shot_end_location':ast.literal_eval,
	'shot_freeze_frame':ast.literal_eval
	}
	
	sb_df = pd.read_csv(path, converters = converter, index_col=0)
	return sb_df

def merge_with_1v1(set_2d_df, set_3d_df, sb_df):
	'''
		Merge 3d and 2d pose data with 1v1 events data
	'''
	set_3d_df = set_3d_df.merge(sb_df, left_on='photo_id', right_index=True, how='left')
	set_2d_df = set_2d_df.merge(sb_df, left_on='photo_id', right_index=True, how='left')
	return set_2d_df, set_3d_df

def straight_bounding_rectangle(points):
    """
    Find the bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    rot_points = points.T.reshape((-1, points.shape[1], points.shape[0]))
    
    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    rval = np.zeros((4, 2))
    rval[0] = [max_x[0], min_y[0]]
    rval[1] = [min_x[0], min_y[0]]
    rval[2] = [min_x[0], max_y[0]]
    rval[3] = [max_x[0], max_y[0]]

    return rval