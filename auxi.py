import re
import ast
import cv2
import math
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mplsoccer import VerticalPitch
import statsmodels.api as sm

import gkpose as gk


def ImageID(df, array_id):
    '''
        Get photo id's of poses
    '''
    return df.loc[array_id, 'file']

def getClusterNames(number_cluster):

	if number_cluster == 4:
		cluster_name = ['Aggressive Set', 'Passive Set', 'Spread', 'Smother']
	else:
		cluster_name = ['Aggressive Set Right', 'Passive Set Right', 'Spread Right', 'Smother Right', 
		'Aggressive Set Left', 'Passive Set Left', 'Spread Left', 'Smother Left']
	return cluster_name

def getClusterSizes(kmeans_preds):

	sizes = np.unique(kmeans_preds, return_counts=True)[1]
	d = {k:v for k,v in enumerate(sizes)}
	return d

# Print aux

def print_full(df, rows=True, columns=False, width=False):
	if rows and columns:
		with pd.option_context('display.max_rows', None, 'display.max_columns', None):
			print(df)
	elif rows:
		with pd.option_context('display.max_rows', None):
			print(df)
	
	elif columns:
		with pd.option_context('display.max_columns', None):
			print(df)
	
	elif width:
		with pd.option_context('max_colwidth', -1):
			print(df)
	else:
		print(df)

def print_cluster_sizes(kmeans_preds, replace_dict):
	'''
		Print cluster sizes
	'''
	print("Cluster sizes: ")
	d = getClusterSizes(kmeans_preds)

	new_d = {replace_dict[k]:v for k,v in d.items()}
	for k in sorted(new_d.keys()):
		print("{}: {}".format(k, new_d[k]), end='\t')
	print()
	return d

def print_cluster_center(closest, replace_dict):
	'''
		Print image that is cluster center
	'''
	print("Closest to cluester center: ")
	new_d = {replace_dict[i]:closest[i] for i in range(len(closest))}
	for k in sorted(new_d.keys()):
		print("{}: {}".format(k, new_d[k]), end='\t')
	print()

def print_penalty_angles(poses_features, kmeans_pens_preds):
	print("Torso Angle, cluster 0", np.mean(poses_features[kmeans_pens_preds == 0][:,0]))
	print("Body Angle, cluster 0", np.mean(poses_features[kmeans_pens_preds == 0][:,4]))

	print("Torso Angle, cluster 1", np.mean(poses_features[kmeans_pens_preds == 1][:,0]))
	print("Body Angle, cluster 1", np.mean(poses_features[kmeans_pens_preds == 1][:,4]))

def print_save_percentage_cluster(good_poses_3d_df, kmeans_pens_preds):
	'''
		Save % for clusters
	'''
	print("Save % for cluster 0 saves:", np.mean(good_poses_3d_df[kmeans_pens_preds == 0]['outcome'] == 'Saved'))
	print("Save % for cluster 1 saves:", np.mean(good_poses_3d_df[kmeans_pens_preds == 1]['outcome'] == 'Saved'))

def printPredictionStats(y_pred, test_df):
	'''
		Prints penalty prediction statistics
	'''

	print('Max xS:', np.max(y_pred))
	print('Min xS:', np.min(y_pred))
	print('Mean xS:', np.mean(y_pred))

	y_pred[y_pred < 0.5] = 0
	# Accuracy
	print("Accuracy:", np.mean(np.array(y_pred) == test_df['outcome']))
	return

# Dict Aux

def player_name_dict():
	d = {
		"Aleksandar Mitrovic": "Aleksandar Mitrović",
		"Alexis Alejandro Sánchez Sánchez": "Alexis Sánchez",
		"Alisson Ramsés Becker": "Alisson",
		"André Ayew": "André Ayew Pelé",
		"Christian Benteke Liolo": "Christian Benteke",
		"Daniel William John Ings": "Danny Ings",
		"David de Gea Quintana": "David de Gea",
		"Dusan Tadic": "Dušan Tadić",
		"Ederson Santana de Moraes":"Ederson", 
		"Gabriel Fernando de Jesus": "Gabriel Jesus",
		"Gylfi Sigurdsson": "Gylfi Sigurðsson",
		"Gylfi Þór Sigurðsson": "Gylfi Sigurðsson",
		"Ilkay Gündogan": "İlkay Gündoğan",
		"James Philip Milner": "James Milner",
		"Jorge Luiz Frello Filho": "Jorginho",
		"Joselu": "José Luis Sanmartín Mato",
		"Joseph Willock": "Joe Willock",
		"Kenedy": "Robert Kenedy Nunes do Nascimento",
		"Kepa Arrizabalaga Revuelta": "Kepa Arrizabalaga",
		"Luka Milivojevic": "Luka Milivojević",
		"Marko Arnautovic": "Marko Arnautović",
		"Raheem Shaquille Sterling": "Raheem Sterling",
		"Raúl Alonso Jiménez Rodríguez": "Raúl Jiménez",
		"Roberto Firmino Barbosa de Oliveira": "Roberto Firmino",
		"Rodrigo": "Rodri",
		"Romelu Lukaku Menama": "Romelu Lukaku",
		"Rui Pedro dos Santos Patrício": "Rui Patrício",
		"Rúben Diogo Da Silva Neves": "Rúben Neves", 
		"Sergio Leonel Agüero del Castillo": "Sergio Agüero",
		"Son Heung-Min": "Son Heung-min",
		"Vicente Guaita Panadero": "Vicente Guaita", 
		"Wayne Mark Rooney": "Wayne Rooney",
		"Willian Borges da Silva": "Willian",
		# "Víctor Camarasa":,

	}
	return d

# import data frames
def import_keypoints(path_2d='data/pose/pose_1v1_2d.csv', path_3d='data/pose/pose_1v1_3d.csv'):
	'''
		Imports 2D and 3D keypoints
	'''
	set_3d_df = pd.read_csv(path_3d, index_col=0)
	set_3d_df = getPhotoID(set_3d_df)

	set_2d_df = pd.read_csv(path_2d, index_col=0)
	set_2d_df = getPhotoID(set_2d_df)
	
	return set_2d_df, set_3d_df

def import_StatsBomb_1v1(path='data/events/1v1_events.csv'):
	'''
		Import StatsBomb 1v1 Data
	'''
	converter = {
	    'location':ast.literal_eval,
	    'shot_end_location':ast.literal_eval,
	    'shot_freeze_frame':ast.literal_eval
	}
	sb_df = pd.read_csv(path, converters=converter, index_col=0)
	sb_df = correct_names(sb_df, 'gk_name')
	return sb_df

def importPenalty3D():
	'''
		Imports penalties 3D poses
	'''

	pose_3d_df = pd.read_csv('data/pose/pen_pose_3d_19_20_20_21.csv', index_col=0)
	pose_3d_2_df = pd.read_csv('data/pose/pen_pose_3d_17_18_18_19.csv', index_col=0)
	joined_pose_3d_df, pose_arr = gk.cleanPenDataFrames(pose_3d_df, pose_3d_2_df)
	return joined_pose_3d_df, pose_arr

def importPenalty2D():
	'''
		Imports penalties 2D poses
	'''
	pose_2d_df = pd.read_csv('data/pose/pen_pose_2d_19_20_20_21.csv', index_col=0)
	pose_2d_2_df = pd.read_csv('data/pose/pen_pose_2d_17_18_18_19.csv', index_col=0)
	joined_pose_2d_df, pose_2d_arr = gk.cleanPenDataFrames(pose_2d_df, pose_2d_2_df)
	return joined_pose_2d_df, pose_2d_arr


# imports

def importImage(img):
    #Import image
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    return image


# DF alter
def cluster_correspondence(kmeans_preds, set_3d_cvi_clean_df, cluster_name):
	'''
	'''
	print("Cluster Correspondence")
	print(len(cluster_name))

	df = make_kmeans_df(kmeans_preds, set_3d_cvi_clean_df)

	if len(cluster_name) == 4:
		gt = pd.read_csv("data/events/cluster_1v1_4.csv")
		m = gt.merge(df, on='img_id', suffixes=['_gt', '_n'])
		g = m.groupby(by=['cluster_gt', 'cluster_n']).count().reset_index(col_level=1)
		g[['sum_gt', 'sum_n']] = 0, 0
		for i, row in g.iterrows():
			c_gt = row['cluster_gt']
			g.at[i, 'sum_gt'] = g.loc[g['cluster_gt'] == c_gt, 'img_id'].sum()
			g.at[i, 'sum_n'] = g.loc[g['cluster_n'] == row['cluster_n'], 'img_id'].sum()

		g['n_gt'], g['gt_n'] = g['img_id']/g['sum_gt'], g['img_id']/g['sum_n']
		print(g)
		a = g.sort_values(by=['n_gt', 'gt_n'], ascending=False).reset_index(drop=True)
		n_gt = a.groupby('cluster_gt')['cluster_n'].apply(list).to_dict()
		s_n_gt = {k:v[0] for k,v in n_gt.items()}
		# print(n_gt)

		a = g.sort_values(by=['gt_n', 'n_gt'], ascending=False).reset_index(drop=True)
		gt_n = a.groupby('cluster_gt')['cluster_n'].apply(list).to_dict()
		s_gt_n = {k:v[0] for k,v in gt_n.items()}		
		# print(gt_n)

		if s_n_gt==s_gt_n:
			return dict((v,k) for k,v in s_gt_n.items())
		else:
			print("Equal??", s_n_gt==s_gt_n)
			print("Equal??", n_gt==gt_n)
			return dict((v,k) for k,v in s_gt_n.items())
	else:
		a = 1

	return


def merge_with_1v1(set_2d_df, set_3d_df, sb_df):
	'''
		Merge 3d and 2d pose data with 1v1 events data
	'''
	set_3d_df = set_3d_df.merge(sb_df, left_on='photo_id', right_index=True, how='left')
	set_2d_df = set_2d_df.merge(sb_df, left_on='photo_id', right_index=True, how='left')
	return set_2d_df, set_3d_df

def createViewInvariant_df(set_3d_df, sets_3d_cvi):
	'''
		Create the view-invariant dataframe
	'''
	set_3d_cvi_df = pd.DataFrame(sets_3d_cvi)
	set_3d_cvi_df.columns = set_3d_cvi_df.columns.astype(str)
	cols = ['file', 'photo_id', 'under_pressure', 'shot_outcome_name', 'distance_to_goal', 'shot_angle', 'gk_name', 'gk_engage']
	set_3d_cvi_df[cols] = set_3d_df[cols]
	return set_3d_cvi_df

def create_GodPoseFeatDf(poses_features, good_poses_3d_df):
	'''
		Create dataframe of good poses features
	'''
	good_poses_feat_df = pd.DataFrame({'outcome': good_poses_3d_df['outcome']})
	good_poses_feat_df = pd.concat([good_poses_feat_df, pd.DataFrame(poses_features, index=good_poses_feat_df.index)], axis=1)

	# Drops off target strikes 
	good_poses_feat_df = good_poses_feat_df.loc[good_poses_feat_df['outcome']  != 'Off T',:]
	good_poses_feat_df.columns = ['outcome','torso_angle','body_height','forward_step', 'hand_height','body_angle']

	# Make target variable boolean - 1=Save, 0=Goal
	good_poses_feat_df['outcome'] = np.array((good_poses_feat_df['outcome'] == 'Saved').astype(int))

	return good_poses_feat_df

def create3D_2D_projection_df(sets_3d_cvi_clean, number_dimensions=2):
	'''
		Create 3D - 2D projection dataset
	'''
	if number_dimensions == 2:
		to_delete = np.array([ x-1 for x in range(0,49) if x%3==0][1:])
		sets_2d_proj = np.delete(sets_3d_cvi_clean, to_delete, 1)
	else:
		sets_2d_proj = sets_3d_cvi_clean.copy()
	return sets_2d_proj

def createTSNEdf(pose_tsne, kmeans_preds, replace_dict=None):
	'''
		Creates a TSNE df
	'''
	df = pd.DataFrame(list(zip(pose_tsne[:, 0], pose_tsne[:, 1], kmeans_preds)),columns =['x', 'y', 'cluster'])
	if replace_dict:
		df = df.replace({"cluster": replace_dict})
	return df

def make_kmeans_df(kmeans_preds, set_3d_cvi_clean_df, cluster_names=None):

	l = [ImageID(set_3d_cvi_clean_df, i) for i in range(len(kmeans_preds))]
	kmeans_dict = {"img_id": l, "cluster": kmeans_preds}
	df = pd.DataFrame.from_dict(kmeans_dict)

	return df

def create_kmeans_df(kmeans_preds, set_3d_cvi_clean_df, cluster_names=None, save=False):
	'''
		Df: image_id -> cluster
	'''
	l = [ImageID(set_3d_cvi_clean_df, i) for i in range(len(kmeans_preds))]
	kmeans_dict = {"img_id": l, "cluster": kmeans_preds}
	df = pd.DataFrame.from_dict(kmeans_dict)

	if cluster_names:
		clusters = list(range(len(cluster_names)))
		replace_dict = {k: v for k,v in zip(clusters, cluster_names)}
		df = df.replace({"cluster": replace_dict})

	if save:
		df.to_csv("data/events/cluster_1v1_4.csv", index=False)
	return df

def clean_train_test(train_df, test_df):
	'''
		Clean train/test sets
	'''

	train_df = train_df.drop('photo_id', axis=1)
	test_df = test_df.drop('photo_id', axis=1)
	train_df = train_df.drop('gk_name', axis=1)
	test_df = test_df.drop('gk_name', axis=1)

	return train_df, test_df

def get_training_test_sets(train_df, test_df):
	'''
		Get training and testing set X and y
	'''
	# Get training set X and y
	y_train = train_df.pop('shot_outcome_name')
	X_train = train_df.values
	
	# Get test set X and y
	y_test = test_df.pop('shot_outcome_name')
	X_test = test_df.values
	
	print("Training Set Size:", len(X_train))
	print("Test Set Size:", len(X_test))
	return train_df, test_df, X_train, y_train, X_test, y_test

# Array IDs

def getPhotoID(df):
    '''
    	Extract photo_id
    '''
    photo_id = []
    for i in range(len(df)):
        photo_id.append(int(re.findall(r"(\d+).", df['file'][i])[0]))
    df['photo_id'] = photo_id
    return df

def getArrayID(pose_df, photo_id):
	'''
		Gets the photo_id array from a dataframe
	'''
	return np.where(np.array(pose_df.index) == photo_id)[0][0]

def getImageID(pose_df, array_id):
    #Input: pose_df - dataframe with raw pose information - index matches to photo name
    #Input: array_id - location of pose in array
    #Returns: photo name/id
    return np.array(pose_df.index)[array_id]

# Else

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


def get_playernames_set(df):
	
	print("Player names: ")
	player_set = set(df['gk_name'].unique().tolist())
	print(sorted(player_set))
	print("___"*20)

def correct_names(df, column):
	df[column] = df[column].replace(player_name_dict())
	return df


if __name__ == '__main__':
	a = 1

	df = pd.read_csv("data/events/1v1_events.csv", index_col=0)
	df = correct_names(df, 'gk_name')
	# print(df)