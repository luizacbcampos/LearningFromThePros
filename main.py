# main

# Imports
import argparse
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

import auxi
import plots
import gkpose as gk

#Defines body pose skeleton for plots
mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], [6, 8], [8, 9]]


def parse_args():

    parser = argparse.ArgumentParser(description='.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-sh', '--show', type=int, default=0, help='Show plots')
    
    parser.add_argument('-d', '--debug', action='store_true', help='Desligar o monitoramento do WandB.' )
    parser.add_argument('-ds', '--dataset', default='synth1', choices=['concrete', 'synth1', 'synth2'], 
    	help='Dataset que sera usado no treino.')
    parser.add_argument('-n_i', '--n_ind', type=int, default=7, help='Tamanho máximo do indivíduo.')
    parser.add_argument('-n_p', '--n_pop', type=int, default=50, help='Tamanho da população.')
    parser.add_argument('-n_g', '--n_gen', type=int, default=50, help='Quantidade de gerações.')
    parser.add_argument('-pc', '--prob_cross', type=float, default=0.9, help='Probabilidade de cruzamento.')
    parser.add_argument('-pm', '--prob_mutation', type=float, default=0.05, help='Probabilidade de mutação.')

    parser.add_argument('-t', '--tournament_size', type=int, default=2, help='Tamanho do torneio.')
    parser.add_argument('-e', '--elitism', type=int, default=1, choices=[0,1], help='Usar elitismo.')


    args = parser.parse_args()
    return args

def show_args(args):
	print("Arguments for run:")
	print('\tSeed:', args.seed)
	print('\tDataset:', args.dataset)
	print('\tMax size individual:', args.n_ind)
	print('\tPopulation size:', args.n_pop)
	print('\tNumber of generations:', args.n_gen)
	print('\tCrossover probability:', args.prob_cross)
	print('\tMutation probability:', args.prob_mutation)
	print('\tTournament Size:', args.tournament_size)
	print('\tElitism:', args.elitism)


# Import and Prepare Data - One on Ones

def import_and_prepare():
	'''
		Import and Prepare Data - One on Ones
	'''
	#Import 2D and 3D pose keypoints
	set_2d_df, set_3d_df = auxi.import_keypoints(path_2d='data/pose/pose_1v1_2d.csv', path_3d='data/pose/pose_1v1_3d.csv')

	#Import StatsBomb 1v1 Data
	sb_df = auxi.import_StatsBomb_1v1(path='data/events/1v1_events.csv')

	#Merge 3d and 2d pose data with 1v1 events data
	set_2d_df, set_3d_df = auxi.merge_with_1v1(set_2d_df, set_3d_df, sb_df)
	
	#Store the 3d and 2d pose coordinates separately
	sets_3d = set_3d_df.loc[:,'0':'47'].values
	sets_2d = set_2d_df.loc[:,'0':'31'].values
	
	return set_2d_df, set_3d_df, sets_2d, sets_3d


# View-Invariance
def viewInvariance(sets_3d, set_3d_df, args):


	#Get camera-view invariant dataset of 3d poses
	cvi_arr = gk.cameraInvariantDataset(sets_3d)
	sets_3d_cvi = gk.flipBehindPoses(cvi_arr)
	
	# Create the view-invariant dataframe and array
	set_3d_cvi_df = auxi.createViewInvariant_df(set_3d_df, sets_3d_cvi)
	
	# Create view-invariant array with GKEM included
	keep_cols = np.array(list(range(48)) + ['gk_engage'])
	sets_3d_cvi = set_3d_cvi_df.loc[:,keep_cols].values
	
	# Camera-view invariance example
	if args.show:
		plots.plot_camera_view_invariance(sets_3d, set_3d_df, sets_3d_cvi, pose_id=319, path='images/1v1_images/')

	# Clean Predictions
	sets_3d_cvi_clean, set_3d_cvi_clean_df = gk.cleanPredictions(set_3d_cvi_df)

	return sets_3d_cvi_clean, set_3d_cvi_clean_df


# Learning Save Technique - Unsupervised Learning

def LearningSaveTechnique(sets_3d_cvi_clean, set_3d_cvi_clean_df, args):
	'''
		Learning Save Technique - Unsupervised Learning
	'''
	
	# Create 3D - 2D projection dataset
	to_delete = np.array([ x-1 for x in range(0,49) if x%3==0][1:])
	sets_2d_proj = np.delete(sets_3d_cvi_clean, to_delete, 1)
	
	# Train K-Means 
	kmeans = KMeans(n_clusters=4, random_state=689).fit(sets_2d_proj)

	# Get cluster membership label for each save - represents chosen save technique
	kmeans_preds = kmeans.predict(sets_2d_proj)
	
	# Clusters are named using domain knowledge
	cluster_name = ['Aggressive Set', 'Passive Set', 'Spread', 'Smother']
	
	auxi.print_cluster_sizes(kmeans_preds, cluster_name)

	#Get 2D TSNE representation of body pose (1244)
	pose_tsne = TSNE(n_components=2, random_state=1445).fit_transform(sets_2d_proj)
	if args.show:
		plots.plotTSNE(pose_tsne, kmeans_preds, cluster_name)

	#Find saves that are closest to cluster centres
	closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sets_2d_proj)
	auxi.print_cluster_center(closest, cluster_name)

	#Plot the most representative saves for each cluster
	if args.show:
		plots.plot_cluster(sets_3d_cvi_clean, set_3d_cvi_clean_df, closest, cluster_name, path='images/1v1_images/', show=args.show)

	return kmeans_preds


# 1v1 Expected Saves Model

def ExpectedSavesModel_1v1(set_3d_cvi_clean_df, args):
	'''
		1v1 Expected Saves Model
	'''
	# Get data for xS model
	df = set_3d_cvi_clean_df.loc[:,'file':]
	df['cluster'] = kmeans_preds #Add save technique as feature
	
	# Get train/test split
	np.random.seed(3615)
	train_df, test_df = gk.getTrainTest(df, test_size=0.3)
	
	# Clean train/test sets
	train_gk_name, train_photo_id = train_df['gk_name'], train_df['photo_id']
	test_gk_name, test_photo_id = test_df['gk_name'], test_df['photo_id']
	train_df, test_df = auxi.clean_train_test(train_df, test_df)

	# Scale the numerical features by removing mean and scaling to unit variance
	features_to_scale = ['shot_angle','distance_to_goal']
	scaler = StandardScaler().fit(train_df[features_to_scale])
	train_df[features_to_scale] = scaler.transform(train_df[features_to_scale])
	test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

	#Get training and testing set X and y	
	train_df, test_df, X_train, y_train, X_test, y_test = auxi.get_training_test_sets(train_df, test_df)
	print("Saved %: ", np.mean(y_train == 1))


	parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10, 100]}
	svm = GridSearchCV(SVC(probability=True), param_grid=parameters, cv=5, scoring='accuracy').fit(X_train, y_train)
	print("Best Parameter Set:", svm.best_params_)
	print("Test Set Accuracy:", np.mean(svm.predict(X_test) == np.array(y_test))*100)

	#Calculate xS map when striker is not under pressure
	xs_map = gk.getXSMap(train_df, svm, scaler, num_clusters=4, up=0)

	#Calculate xS map for when striker is under pressure
	xs_map_up = gk.getXSMap(train_df, svm, scaler, num_clusters=4, up=1)

	if args.show:
		cluster_name = ['Aggressive Set', 'Passive Set', 'Spread', 'Smother']
		plots.plotDoubleXSMap(xs_map, xs_map_up, cluster_name)

		#Optimal technique map
		plots.plotBestTechniqueUp(xs_map, xs_map_up, cluster_name)

	return train_gk_name, test_gk_name, train_df, test_df, svm


# Pro Goalkeeper Scouting

def proGoalkeeperScouting(train_gk_name, test_gk_name , train_df, test_df, svm, args):
	'''
		Pro Goalkeeper Scouting
	'''
	#Reset the index of test set
	test_df.reset_index(drop=True, inplace=True)
	
	#Derive the optimal save technique for each save
	test_df['optimal_cluster'] = gk.getProsOptimalCluster(test_df, svm, num_clusters=4)
	train_df['optimal_cluster'] = gk.getProsOptimalCluster(train_df, svm, num_clusters=4)

	#The chosen technique for each save
	test_df['chosen_cluster'] = np.argmax(test_df.filter(regex='cluster_').values, axis=1)
	train_df['chosen_cluster'] = np.argmax(train_df.filter(regex='cluster_').values, axis=1)

	#Rank Pro Goalkeepers by their % correct usage of save technique
	test_df['gk_name'], train_df['gk_name'] = np.array(test_gk_name), np.array(train_gk_name)
	gk_df = pd.concat([train_df[['gk_name','chosen_cluster','optimal_cluster']], test_df[['gk_name','chosen_cluster','optimal_cluster']]])
	print("Percentage optimal:", np.mean(gk_df['chosen_cluster'] == gk_df['optimal_cluster']) * 100)


	gk_df['correct_cluster'] = (gk_df['chosen_cluster'] == gk_df['optimal_cluster'])
	gk_ranking = gk_df[['gk_name','correct_cluster']].groupby(['gk_name'], as_index=False).mean()
	gk_ranking['shots_faced'] = np.array(gk_df.groupby(['gk_name'], as_index=False).count()['correct_cluster'])
	gk_ranking.sort_values(by=['correct_cluster'], ascending=False, inplace=True)
	gk_ranking.reset_index(drop=True, inplace=True)

	#Ranking of Pro Keepers with >= 15 1v1s faced
	print(gk_ranking[gk_ranking['shots_faced'] >= 15].reset_index(drop=True))	
	return


# Penalty Analysis

def PenaltyAnalysis(args):

	#3D pose data
	joined_pose_3d_df, pose_arr = auxi.importPenalty3D()
	# 2D pose data
	joined_pose_2d_df, pose_2d_arr = auxi.importPenalty2D()

	# Percentage of pens that were saved in our dataset
	print("Percentage of saved penalties:", np.mean(joined_pose_3d_df['outcome'] == 'Saved') * 100)

	if args.show:
		# Show image, image with 2D pose overlay, and 3D pose estimate
		photo_id = 315
		plots.plot_pose_estimation(joined_pose_3d_df, pose_arr, pose_2d_arr, photo_id, show=args.show)
		
		pic_ids, path = [388, 20, 3, 243, 302, 377], 'images/pen_images/combined_data/'
		plots.plot_penalty_examples(pose_arr, joined_pose_3d_df, pic_ids, path, show=args.show)
		
	# Get camera-view invariant dataset of 3d poses
	pen_pose_vi = gk.cameraInvariantDataset(pose_arr)
	
	# Rotates the poses from images taken from behind by 180 degrees
	pen_pose_vi = gk.flipBehindPoses(pen_pose_vi)

	# Good Poses DataFrame
	good_poses_3d_df = gk.cleanPenPredictions(joined_pose_3d_df)

	# Good Poses Matrix
	good_poses_3d_arr = good_poses_3d_df.loc[:,'0':].values
	
	# Convert all the good poses to the features space
	poses_features = gk.PenFeatureSpace(good_poses_3d_arr)
	
	# Fit K-Means model
	kmeans_pens = KMeans(n_clusters=2, random_state = 13).fit(poses_features)
	kmeans_pens_preds = kmeans_pens.predict(poses_features)
	auxi.print_cluster_sizes(kmeans_pens_preds, ['Cluster 0', 'Cluster 1'])
	
	auxi.print_penalty_angles(poses_features, kmeans_pens_preds)


	# TSNE representation of body pose
	pens_tsne = TSNE(n_components=2, random_state=29).fit_transform(poses_features)
	
	if args.show:
		plots.plotTSNE(pens_tsne, kmeans_pens_preds, ['Cluster 0', 'Cluster 1'], number=2)

		# GMM - 3D pose, 2D pose viz cluster examples
		ax_array, path = [1, 5, 9, 13, 17], 'images/pen_images/combined_data/'
		plots.penalty_clusterExamples(good_poses_3d_arr, good_poses_3d_df, kmeans_pens_preds, ax_array, path, show=args.show)

	# Save % for clusters
	auxi.print_save_percentage_cluster(good_poses_3d_df, kmeans_pens_preds)

	# Create dataframe of good poses features
	good_poses_feat_df = auxi.create_GodPoseFeatDf(poses_features, good_poses_3d_df)
	continuous_var = ['torso_angle','body_height','forward_step','hand_height','body_angle']
	
	# Train/Test Split (70/30)
	split_index = good_poses_feat_df.index[int(len(good_poses_feat_df)*0.7)]
	test_df = good_poses_feat_df.loc[split_index:, :].copy() 
	train_df = good_poses_feat_df.loc[:split_index-1, :].copy() 

	#Standardise continuous variables
	scaler = StandardScaler()
	scaler.fit(train_df[continuous_var])
	train_df[continuous_var] = scaler.transform(train_df[continuous_var])
	test_df[continuous_var] = scaler.transform(test_df[continuous_var])
	
	# Add intercept term
	train_df['coef'], test_df['coef'] = 1, 1

	# Train logistic regression
	log_reg = sm.Logit(train_df['outcome'], train_df[train_df.columns[1:]]).fit()
	# Logistic regression summary
	print(log_reg.summary())

	# Predictions
	y_pred = log_reg.predict(test_df[test_df.columns[1:]])
	# Prediction stats
	auxi.printPredictionStats(y_pred, test_df)

	return



if __name__ == '__main__':
	args = parse_args()

	# Import and Prepare Data - One on Ones
	set_2d_df, set_3d_df, sets_2d, sets_3d = import_and_prepare()

	# View-Invariance
	sets_3d_cvi_clean, set_3d_cvi_clean_df = viewInvariance(sets_3d, set_3d_df, args)

	# Learning Save Technique - Unsupervised Learning
	kmeans_preds = LearningSaveTechnique(sets_3d_cvi_clean, set_3d_cvi_clean_df, args)
	
	# 1v1 Expected Saves Model
	train_gk_name, test_gk_name, train_df, test_df, svm = ExpectedSavesModel_1v1(set_3d_cvi_clean_df, args)

	# Pro Goalkeeper Scouting
	proGoalkeeperScouting(train_gk_name, test_gk_name , train_df, test_df, svm, args)

	# Penalty Analysis
	PenaltyAnalysis(args)