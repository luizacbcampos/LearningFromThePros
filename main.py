# main

# Imports
import re
import ast
import cv2
import math
import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from mplsoccer import VerticalPitch
import statsmodels.api as sm

import plots
import gkpose as gk

# Import and Prepare Data - One on Ones

#Import 3D pose keypoints
set_3d_df = pd.read_csv('data/pose/pose_1v1_3d.csv', index_col=0)
set_3d_df = gk.getPhotoID(set_3d_df)

#Import 2D pose keypoints (given in image coordinates)
set_2d_df = pd.read_csv('data/pose/pose_1v1_2d.csv', index_col=0)
set_2d_df = gk.getPhotoID(set_2d_df)

#Import StatsBomb 1v1 Data
converter = {
    'location':ast.literal_eval,
    'shot_end_location':ast.literal_eval,
    'shot_freeze_frame':ast.literal_eval
}
sb_df = pd.read_csv('data/events/1v1_events.csv', converters = converter, index_col=0)

#Merge 3d and 2d pose data with 1v1 events data
set_3d_df = set_3d_df.merge(sb_df, left_on='photo_id', right_index=True, how='left')
set_2d_df = set_2d_df.merge(sb_df, left_on='photo_id', right_index=True, how='left')

#Store the 3d and 2d pose coordinates separately
sets_3d = set_3d_df.loc[:,'0':'47'].values
sets_2d = set_2d_df.loc[:,'0':'31'].values

# View-Invariance

#Get camera-view invariant dataset of 3d poses
cvi_arr = gk.cameraInvariantDataset(sets_3d)
sets_3d_cvi = gk.flipBehindPoses(cvi_arr)

#Create the view-invariant dataframe
set_3d_cvi_df = pd.DataFrame(sets_3d_cvi)
set_3d_cvi_df.columns = set_3d_cvi_df.columns.astype(str)
cols = ['file', 'photo_id', 'under_pressure', 'shot_outcome_name', 'distance_to_goal', 'shot_angle', 'gk_name', 'gk_engage']
set_3d_cvi_df[cols] = set_3d_df[cols]

#Create view-invariant array with GKEM included
keep_cols = np.array(list(range(48)) + ['gk_engage'])
sets_3d_cvi = set_3d_cvi_df.loc[:,keep_cols].values

#Defines body pose skeleton for plots
mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], [6, 8], [8, 9]]

#Camera-view invariance example
plots.plot_camera_view_invariance(sets_3d, set_3d_df, sets_3d_cvi, pose_id=319, path='images/1v1_images/')

sets_3d_cvi_clean, set_3d_cvi_clean_df = gk.cleanPredictions(set_3d_cvi_df)

# Learning Save Technique - Unsupervised Learning


#Create 3D - 2D projection dataset
to_delete = np.array([ x-1 for x in range(0,49) if x%3==0][1:])
sets_2d_proj = np.delete(sets_3d_cvi_clean, to_delete, 1)

#Train K-Means 
kmeans = KMeans(n_clusters=4, random_state = 689).fit(sets_2d_proj)

#Get cluster membership label for each save - represents chosen save technique
kmeans_preds = kmeans.predict(sets_2d_proj)
np.unique(kmeans_preds, return_counts=True)[1]

#Clusters are named using domain knowledge
cluster_name = ['Aggressive Set', 'Passive Set', 'Spread', 'Smother']

#Get 2D TSNE representation of body pose (1244)
pose_tsne = TSNE(n_components=2, random_state=1445).fit_transform(sets_2d_proj)

plots.plotTSNE(pose_tsne, kmeans_preds, cluster_name)

#Find saves that are closest to cluster centres
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sets_2d_proj)

#Plot the most representative saves for each cluster
plots.plot_cluster(sets_3d_cvi_clean, set_3d_cvi_clean_df, closest, cluster_name, path='images/1v1_images/')

# 1v1 Expected Saves Model

#Get data for xS model
df = set_3d_cvi_clean_df.loc[:,'file':]
df['cluster'] = kmeans_preds #Add save technique as feature

#Get train/test split
np.random.seed(3615)
train_df, test_df = gk.getTrainTest(df, test_size=0.3)

#Clean train/test sets
train_gk_name = train_df['gk_name']
train_photo_id = train_df['photo_id']
test_gk_name = test_df['gk_name']
test_photo_id = test_df['photo_id']

train_df.drop('photo_id', axis=1, inplace=True)
test_df.drop('photo_id', axis=1, inplace=True)
train_df.drop('gk_name', axis=1, inplace=True)
test_df.drop('gk_name', axis=1, inplace=True)

#Scale the numerical features by removing mean and scaling to unit variance
features_to_scale = ['shot_angle','distance_to_goal']
scaler = StandardScaler().fit(train_df[features_to_scale])
train_df[features_to_scale] = scaler.transform(train_df[features_to_scale])
test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

#Get training set X and y
y_train = train_df.pop('shot_outcome_name')
X_train = train_df.values
#Get test set X and y
y_test = test_df.pop('shot_outcome_name')
X_test = test_df.values
print("Training Set Size:", len(X_train))
print("Test Set Size:", len(X_test))

np.mean(y_train == 1)

parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10, 100]}
svm = GridSearchCV(SVC(probability=True), param_grid=parameters, cv=5, scoring='accuracy').fit(X_train, y_train)
print("Best Parameter Set:", svm.best_params_)
print("Test Set Accuracy:", np.mean(svm.predict(X_test) == np.array(y_test))*100)

#Calculate xS map when striker is not under pressure
xs_map = gk.getXSMap(train_df, svm, scaler, num_clusters=4, up=0)

plots.plotXSMap(xs_map, num_clusters=4, cluster_names=cluster_name)

#Calculate xS map for when striker is under pressure
xs_map_up = gk.getXSMap(train_df, svm, scaler, num_clusters=4, up=1)

plots.plotXSMap(xs_map_up, num_clusters=4, cluster_names=cluster_name)

plots.plotDoubleXSMap(xs_map, xs_map_up, cluster_name)

#Optimal technique map
plots.plotBestTechniqueUp(xs_map, xs_map_up, cluster_name)

# Pro Goalkeeper Scouting

