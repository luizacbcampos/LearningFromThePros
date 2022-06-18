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
	return new_d

def print_cluster_center(set_3d_cvi_clean_df, closest, replace_dict):
	'''
		Print image that is cluster center
	'''
	print("Closest to cluster center: ")
	new_d = {replace_dict[i]:ImageID(set_3d_cvi_clean_df, closest[i]) for i in range(len(closest))}

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

# create DF
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

def create_kmeans_df(kmeans_preds, set_3d_cvi_clean_df, cluster_names=None, save=False):
	'''
		Df: image_id -> cluster
	'''
	df = make_kmeans_df(kmeans_preds, set_3d_cvi_clean_df, cluster_names)

	if cluster_names:
		clusters = list(range(len(cluster_names)))
		replace_dict = {k: v for k,v in zip(clusters, cluster_names)}
		df = df.replace({"cluster": replace_dict})

	if save:
		df.to_csv("data/events/cluster_1v1_4.csv", index=False)
	return df

def make_kmeans_df(kmeans_preds, set_3d_cvi_clean_df, cluster_names=None):

	l = [ImageID(set_3d_cvi_clean_df, i) for i in range(len(kmeans_preds))]
	kmeans_dict = {"img_id": l, "cluster": kmeans_preds}
	df = pd.DataFrame.from_dict(kmeans_dict)

	return df

def create_side_df(sets_2d, set_2d_df, save=False):
	'''
		Creates a DF that shows the side the keepers are inclined to
	'''
	sets_2d, set_2d_df, sets_2d_cvi, set_2d_cvi_df = ViewInvarianceData(sets_2d, set_2d_df, vi=False)
	sets_2d_cvi_clean, set_2d_cvi_clean_df = gk.cleanPredictions(set_2d_cvi_df, s=32)

	sets_2d_cvi_clean = np.delete(sets_2d_cvi_clean, 32, 1)
	sides = []

	for i in range(len(sets_2d_cvi_clean)):
		pose = sets_2d_cvi_clean[i].reshape((16,2))
		side = pick_side(pose)
		sides.append(side)
		# print(i, ImageID(set_2d_cvi_clean_df, i), side)

	l = [ImageID(set_2d_cvi_clean_df, i) for i in range(len(sets_2d_cvi_clean))]
	d = {"img_id": l, "side": sides}
	df = pd.DataFrame.from_dict(d).replace({'side': {0: 'Right', 1: "Left"}})
	if save:
		df.to_csv("data/events/side_1v1.csv", index=False)
	return


# DF alter

def column_proportions(g):
	'''
		Creates proportions from cluster columns
	'''

	g[['sum_gt', 'sum_n']] = 0, 0
	for i, row in g.iterrows():
		g.at[i, 'sum_gt'] = g.loc[g['cluster_gt'] == row['cluster_gt'], 'img_id'].sum() # quantos do gt estão nesse grupo
		g.at[i, 'sum_n'] = g.loc[g['cluster_n'] == row['cluster_n'], 'img_id'].sum() # quantos dos cluster novos estão nesse grupo

	g['perc_gt'] = g['img_id']/g['sum_gt'] # quantidade de poses dv por quantos do gt estão nesse grupo
	g['perc_novo'] = g['img_id']/g['sum_n'] # quantidade de poses dv por quantos dos cluster novos estão nesse grupo
	return g

def cluster_proportions_df(df, gt, on=['img_id'], cols=['cluster_gt', 'cluster_n']):

	m = gt.merge(df, on=on, suffixes=['_gt', '_n'])
	g = m.groupby(by=cols).count().reset_index(col_level=1)
	g = column_proportions(g)
	return g

def cluster_dict(g, by):
	a = g.sort_values(by=by, ascending=False).reset_index(drop=True)
	d = a.groupby('cluster_gt')['cluster_n'].apply(list).to_dict()
	return d

def remove_left_right(d):
	def invert_dict(d): 
	    inverse = dict() 
	    for key in d: # Go through the list that is saved in the dict: 
	        for item in d[key]: # Check if in the inverted dict the key exists
	            if item not in inverse: # If not create a new list
	                inverse[item] = [key] 
	            else: 
	                inverse[item].append(key) 
	    return inverse

	og = ['Aggressive Set', 'Passive Set', 'Spread', 'Smother']
	dicio = {k: [v] for k,v in d.items()}

	inverse = invert_dict(dicio)
	for k,v in inverse.items():
		if len(v) > 1:
			new_list = set([g.split(" Left")[0].split(' Right')[0] for g in v])
			if len(new_list) == 1:
				inverse[k] = list(new_list)
	
	inverse = invert_dict(inverse)
	dicio = {k:v[0] for k,v in inverse.items()}
	return dicio

def cluster_correspondence(kmeans_preds, set_3d_cvi_clean_df, cluster_name):
	'''
	'''
	print("Cluster Correspondence")
	print(len(cluster_name))

	df = make_kmeans_df(kmeans_preds, set_3d_cvi_clean_df)
	gt = pd.read_csv("data/events/cluster_1v1_4.csv")

	if len(cluster_name) == 4:
		g = cluster_proportions_df(df, gt, cols=['cluster_gt', 'cluster_n'])
		# print(g)

		perc_gt = cluster_dict(g, by=['perc_gt', 'perc_novo'])
		s_perc_gt = {k:v[0] for k,v in perc_gt.items()}
		# print(perc_gt)

		perc_novo = cluster_dict(g, by=['perc_novo', 'perc_gt'])
		s_perc_novo = {k:v[0] for k,v in perc_novo.items()}		
		# print(perc_novo)

		if s_perc_gt==s_perc_novo:
			return dict((v,k) for k,v in s_perc_novo.items())
		else:
			print("Equal??", s_perc_gt==s_perc_novo)
			print("Equal??", perc_gt==perc_novo)
			return dict((v,k) for k,v in s_perc_novo.items())
	else:
		side_df = pd.read_csv("data/events/side_1v1.csv")
		gt = gt.merge(side_df, on='img_id')
		df = df.merge(side_df, on='img_id')
		
		# join cols
		gt["cluster"] = gt["cluster"] + ' ' + gt["side"]

		g = cluster_proportions_df(df, gt, on=['img_id', 'side'], cols=['cluster_gt', 'cluster_n'])
		print(g)
		

		perc_gt = cluster_dict(g, by=['perc_gt', 'perc_novo'])
		s_perc_gt = {k:v[0] for k,v in perc_gt.items()}
		# print(s_perc_gt)

		perc_novo = cluster_dict(g, by=['perc_novo', 'perc_gt'])
		s_perc_novo = {k:v[0] for k,v in perc_novo.items()}		
		# print(s_perc_novo)

		# s_perc_gt = remove_left_right(s_perc_gt)
		# s_perc_novo = remove_left_right(s_perc_novo)
		# print("Equal??", s_perc_gt==s_perc_novo)
		# print("> ", s_perc_gt)

		s_perc_novo = {'Aggressive Set':0, 'Passive Set':1, 'Spread Right':2, 'Smother Right':3, 'Spread Left':4, 'Smother Left':5}
		return dict((v,k) for k,v in s_perc_novo.items())
	return

def merge_with_1v1(set_2d_df, set_3d_df, sb_df):
	'''
		Merge 3d and 2d pose data with 1v1 events data
	'''
	set_3d_df = set_3d_df.merge(sb_df, left_on='photo_id', right_index=True, how='left')
	set_2d_df = set_2d_df.merge(sb_df, left_on='photo_id', right_index=True, how='left')
	return set_2d_df, set_3d_df

def ViewInvarianceData(sets_, set_df, vi):

	#Get camera-view invariant dataset of 3d poses
	cvi_arr = gk.cameraInvariantDataset(sets_, vi=vi)
	sets_cvi = gk.flipBehindPoses(cvi_arr)
	
	# Create the view-invariant dataframe and array
	set_cvi_df = createViewInvariant_df(set_df, sets_cvi)
	
	# Create view-invariant array with GKEM included
	keep_cols = np.array(list(range(sets_.shape[1])) + ['gk_engage'])
	sets_cvi = set_cvi_df.loc[:,keep_cols].values

	return sets_, set_df, sets_cvi, set_cvi_df



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

def pick_side(points):
	'''
		Find the side the goalkeeper is inclinated (front-view, as if we're facing them)
		Returns the side (0: right, 1: left)
	'''
	x_middle, y_middle = straight_bounding_rectangle(points).mean(axis=0)
	s = points.shape

	points = np.delete(points, 1, 1).reshape((s[0],))
	# print(x_middle, points)
	coluna = points[[6,8,9]]
	x_middle = coluna.mean()
	# print(x_middle, coluna.mean())
	left_side, right_side = len(points[(points < x_middle*0.95)]), len(points[points > x_middle*1.05])
	# print(left_side, right_side)
	if left_side > right_side:
		return 1
	else:
		return 0

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