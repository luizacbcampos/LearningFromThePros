# main
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Imports
import os
import wandb
import argparse
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min, confusion_matrix, f1_score, recall_score, precision_score

from imblearn.under_sampling import NearMiss

import statsmodels.api as sm

import auxi
import plots
import gkpose as gk

#Defines body pose skeleton for plots
mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], [6, 8], [8, 9]]


def parse_args():

    parser = argparse.ArgumentParser(description='.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--debug', action='store_true', help='Desligar o monitoramento do WandB.' )
    parser.add_argument('-s', '--save', type=int, default=0, help='Save plots')
    parser.add_argument('-sh', '--show', type=int, default=0, help='Show plots')
    parser.add_argument('-v1', '--view_invariant1', type=int, default=1, help='Use view-invariance on 1v1')
    parser.add_argument('-nd', '--number_dimensions', type=int, default=2, help='Number of dimensions to use')
    parser.add_argument('-si', '--split_sides', type=int, default=0, help='Split saves into left and right')
    parser.add_argument('-g', '--grid_search', type=int, default=0, help='Add more parameters to GridSearchCV')
    parser.add_argument('-v2', '--view_invariant2', type=int, default=1, help='Use view-invariance on penalties')
    parser.add_argument('-p', '--penalty_location', type=int, default=0, help='Use penalty location')
    parser.add_argument('-ha', '--hand', type=int, default=0, help='Add dominant hand parameter')
    parser.add_argument('-he', '--height', type=int, default=0, help='Add goalkeeper height information')


    args = parser.parse_args()
    return args

def show_args(args):
	print("Arguments for run:")
	print('\tView Invariance on 1v1:', args.view_invariant1)
	print('\tNumber of Dimensions:', args.number_dimensions)
	print('\tSplit Sides:', args.split_sides)
	print('\tGrid Search:', args.grid_search)
	print('\tView Invariance on penalties:', args.view_invariant2)
	print('\tPenalty Location:', args.penalty_location)
	print('\tDominant Hand:', args.hand)
	print('\tGoalkeeper Height:', args.height)
	print('\tSave:', args.save)
	print('\tShow:', args.show)


# Import and Prepare Data - One on Ones

def import_and_prepare():
	'''
		Import and Prepare Data - One on Ones
		Factors: None
		Metrics: none
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
	'''
		View Invariance
		Factors: view_invariant1
		Metrics: None
	'''


	# Get camera-view invariant data
	sets_3d, set_3d_df, sets_3d_cvi, set_3d_cvi_df = auxi.ViewInvarianceData(sets_3d, set_3d_df, vi=args.view_invariant1)
	
	# Camera-view invariance example
	if args.show or args.save:
		plots.plot_camera_view_invariance(sets_3d, set_3d_df, sets_3d_cvi, pose_id=319, path='images/1v1_images/', show=args.show)

	# Clean Predictions
	sets_3d_cvi_clean, set_3d_cvi_clean_df = gk.cleanPredictions(set_3d_cvi_df)
	return sets_3d_cvi_clean, set_3d_cvi_clean_df


# Learning Save Technique - Unsupervised Learning

def LearningSaveTechnique(sets_3d_cvi_clean, set_3d_cvi_clean_df, args, start='\t>'):
	'''
		Learning Save Technique - Unsupervised Learning
		Factors: view_invariant1, number_dimensions, split_sides
		Metrics: Dividir a grande área em quadrados e observar a variação da melhor técnica por quadrado;
	'''
	def preKmeans(proj_set):
		if not args.split_sides:
			return KMeansCalc(proj_set)
		else:
			# number_cluster = 6
			
			side_df = pd.read_csv("data/events/side_1v1.csv")
			side_df = set_3d_cvi_clean_df.merge(side_df, right_on='img_id', left_on='file')

			left_ids = side_df[side_df['side']=='Left'].index.to_numpy() 
			right_ids = side_df[side_df['side']=='Right'].index.to_numpy()

			
			Lnumber_cluster, Lkmeans, Lkmeans_preds, Lcluster_name, Lclosest = KMeansCalc(proj_set[left_ids])
			Rnumber_cluster, Rkmeans, Rkmeans_preds, Rcluster_name, Rclosest = KMeansCalc(proj_set[right_ids])
			
			
			number_cluster = Lnumber_cluster + Rnumber_cluster
			cluster_name = ['Left '+ i for i in Lcluster_name] + ['Right '+ i for i in Rcluster_name]

			# Join preds
			Rkmeans_preds = Rkmeans_preds+4
			kmeans_preds = np.ones((len(proj_set)))
			kmeans_preds[right_ids] = Rkmeans_preds
			kmeans_preds[left_ids] = Lkmeans_preds
			
			closest = np.concatenate([left_ids[Lclosest], right_ids[Rclosest]]).ravel()
			kmeans = [Lkmeans, Rkmeans]

			return number_cluster, kmeans, kmeans_preds, cluster_name, closest	

	def KMeansCalc(proj_set):

		# Train K-Means 
		kmeans = KMeans(n_clusters=number_cluster, random_state=689).fit(proj_set)

		# Get cluster membership label for each save - represents chosen save technique
		kmeans_preds = kmeans.predict(proj_set)

		# Clusters are named using domain knowledge
		cluster_name = auxi.getClusterNames(number_cluster)

		#Find saves that are closest to cluster centres
		closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, proj_set)
		return number_cluster, kmeans, kmeans_preds, cluster_name, closest

	# Create 3D - 2D projection dataset
	sets_2d_proj = auxi.create3D_2D_projection_df(sets_3d_cvi_clean, number_dimensions=args.number_dimensions)

	number_cluster = 4
	number_cluster, kmeans, kmeans_preds,  cluster_name, closest = preKmeans(sets_2d_proj) #KMeansCalc()
	
	cluster_dict = auxi.cluster_correspondence(kmeans_preds, set_3d_cvi_clean_df, cluster_name)
	print(start, "Cluster Dict:", cluster_dict)
	c_size_d = auxi.print_cluster_sizes(kmeans_preds, cluster_dict, start)
	

	# df = auxi.create_kmeans_df(kmeans_preds, set_3d_cvi_clean_df, cluster_name, save=True)

	# Get 2D TSNE representation of body pose (1244)
	pose_tsne = TSNE(n_components=2, random_state=1445, init='random', learning_rate='auto').fit_transform(sets_2d_proj)
	if args.show:
		plots.plotTSNE(pose_tsne, kmeans_preds, cluster_name, number=len(cluster_name), show=args.show)
	
	c_center = auxi.print_cluster_center(set_3d_cvi_clean_df, closest, cluster_dict, start)

	#Plot the most representative saves for each cluster
	if args.show:
		plots.plot_cluster(sets_3d_cvi_clean, set_3d_cvi_clean_df, closest, cluster_name, path='images/1v1_images/', show=args.show)

	cluster_mean_df = auxi.mean_pose_per_cluster(kmeans_preds, set_3d_cvi_clean_df, cluster_dict, c_center)

	if number_cluster == 8:
		for k,v in cluster_dict.items():
			val = v.split(' Left')[0]
			val = val.split(' Right')[0]
			cluster_dict[k] = val

		new_cluster_dict = {v: [] for v in set(cluster_dict.values())}
		for k, v in cluster_dict.items():
			new_cluster_dict[v].append(k)

		for k, v in new_cluster_dict.items():
			values = sorted(v)
			kmeans_preds = np.where(kmeans_preds == values[1], values[0], kmeans_preds)

		cluster_dict = dict((v[0],k) for k,v in new_cluster_dict.items())

	if not args.debug:
		wandb.config.update({"KMeans Cluster Count": number_cluster})

		TSNE_df = auxi.createTSNEdf(pose_tsne, kmeans_preds, cluster_dict)

		wandb.sklearn.plot_elbow_curve(KMeans(random_state=689).fit(sets_2d_proj), sets_2d_proj)
		wandb.sklearn.plot_silhouette(kmeans, sets_2d_proj, kmeans_preds)

		wandb.log({"KMeans 1v1 Table": auxi.make_kmeans_df(kmeans_preds, set_3d_cvi_clean_df, cluster_dict)})
		wandb_LearningSaveTechnique(TSNE_df, c_size_d, cluster_mean_df)

	baseline = {'Aggressive Set':0, 'Passive Set':1, 'Spread':2 , 'Smother':3}
	kmeans_preds = kmeans_preds + 4
	replace = {k+4:baseline[v] for k,v in cluster_dict.items()}
	for k, v in replace.items():
		kmeans_preds = np.where(kmeans_preds==k, v, kmeans_preds)
	return kmeans_preds

def wandb_LearningSaveTechnique(TSNE_df, c_size_d, cluster_mean_df):

	def TSNE_plot():
		data = TSNE_df.values.tolist()
		table = wandb.Table(data=data, columns = ['t-SNE_1', "t-SNE_2", "cluster"])
		wandb.log({"TSNE 1v1" : wandb.plot.scatter(table, 't-SNE_1', 't-SNE_2')})

	def cluster_sizes():

		data = [[label, val] for label, val in c_size_d.items()]
		table = wandb.Table(data=data, columns = ["cluster", "count"])
		wandb.log({"KMeans 1v1 Size" : wandb.plot.bar(table, "cluster", "count", title="Cluster Size in KMeans 1v1")})

	def cluster_mean():
		data = cluster_mean_df.values.tolist()
		table = wandb.Table(data=data, columns = cluster_mean_df.columns.tolist())
		wandb.log({"Cluster Mean 1v1" : table})

	TSNE_plot()
	cluster_sizes()
	cluster_mean()
	return


# 1v1 Expected Saves Model

def ExpectedSavesModel_1v1(set_3d_cvi_clean_df, kmeans_preds, args, start='\t>'):
	'''
		1v1 Expected Saves Model
		Factors: grid_search
		Metrics:  Acuracy, Precision, Recall, F1-Score
	'''
	def grid_search_parameters():
		if args.grid_search:
			parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[0.1, 1, 10, 100]}
		else:
			parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10, 100]}
		return parameters

	# Get data for xS model
	df = set_3d_cvi_clean_df.loc[:,'file':]
	df['cluster'] = kmeans_preds # Add save technique as feature
	number_cluster = len(set(kmeans_preds))

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
	print(start, "Saved %: ", np.mean(y_train == 1))


	parameters = grid_search_parameters()
	svm = GridSearchCV(SVC(probability=True), param_grid=parameters, cv=5, scoring='accuracy').fit(X_train, y_train)
	print(start, "Best Parameter Set:", svm.best_params_)
	
	y_pred = svm.predict(X_test)
	auxi.print_classification_metrics(y_test, y_pred, start)

	#Calculate xS map when striker is not under pressure
	xs_map = gk.getXSMap(train_df, svm, scaler, num_clusters=number_cluster, up=0) #ADD
	
	# best_tech_xs_map = np.argmax(xs_map, axis=0)

	#Calculate xS map for when striker is under pressure
	xs_map_up = gk.getXSMap(train_df, svm, scaler, num_clusters=number_cluster, up=1) #ADD

	cluster_name = ['Aggressive Set', 'Passive Set', 'Spread', 'Smother']
	if args.show:
		plots.plotDoubleXSMap(xs_map, xs_map_up, cluster_name, show=args.show)
		# Optimal technique map
		plots.plotBestTechniqueUp(xs_map, xs_map_up, cluster_name, show=args.show)

	if not args.debug:
		wandb.log({"XS Map": plots.plotBestTechniqueUp(xs_map, xs_map_up, cluster_name, show=args.show)})
		# Save to pc
		run_type = "v1{}-nd{}-g{}-si{}".format(args.view_invariant1, args.number_dimensions, args.grid_search, args.split_sides)
		file_number = len([name for name in os.listdir('./results') if os.path.isfile(name)])/2
		run_type = "{}-{}.npy".format(run_type, file_number)
		np.save("results/xs_map"+run_type, xs_map)
		np.save("results/xs_map_up"+run_type, xs_map_up)

		wandb_ExpectedSavesModel_1v1(svm, X_test, y_test)

	return train_gk_name, test_gk_name, train_df, test_df, svm


def wandb_ExpectedSavesModel_1v1(svm, X_test, y_test):
	'''
		Upload to wandb
	'''
	wandb.config.update(svm.best_params_)
	y_pred = svm.predict(X_test)

	cm = wandb.sklearn.plot_confusion_matrix(y_test, y_pred, normalize='all')
	# cm = wandb.plot.confusion_matrix(probs=None, y_true=y_test, preds=y_pred)
	
	cmatrix, f1, recall, precision, test_set_acc = auxi.classification_metrics(y_test, y_pred)
	
	d = {"conf_mat 1v1": cm, "Accuracy 1v1": test_set_acc, 'F1 1v1': f1, "Recall 1v1": recall, "Precision 1v1": precision}
	wandb.log(d)
	
	return

# Pro Goalkeeper Scouting

def proGoalkeeperScouting(train_gk_name, test_gk_name , train_df, test_df, svm, args):
	'''
		Pro Goalkeeper Scouting
		Metrics: Mean Average Precision (original list as ground-truth); Spearman’s ρ

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

	gk_df['correct_cluster'] = (gk_df['chosen_cluster'] == gk_df['optimal_cluster'])
	gk_ranking = gk_df[['gk_name','correct_cluster']].groupby(['gk_name'], as_index=False).mean()
	gk_ranking['shots_faced'] = np.array(gk_df.groupby(['gk_name'], as_index=False).count()['correct_cluster'])
	gk_ranking.sort_values(by=['correct_cluster'], ascending=False, inplace=True)
	gk_ranking.reset_index(drop=True, inplace=True)

	# Ranking of Pro Keepers with >= 15 1v1s faced
	ranking = gk_ranking[gk_ranking['shots_faced'] >= 15].reset_index(drop=True)

	# Show metrics
	auxi.print_ranking_metrics(gk_df, gk_ranking, ranking, start='\t>')
	print("> Ranking of Pro Keepers with >= 15 1v1s faced")
	print(ranking)



	if not args.debug:
		wandb_proGoalkeeperScouting(gk_df, gk_ranking)		
	return

def wandb_proGoalkeeperScouting(gk_df, gk_ranking):
	'''
		Upload to wandb
	'''
	
	ranking = gk_ranking[gk_ranking['shots_faced'] >= 15].reset_index(drop=True)
	opt_perc, mAP, rho, pval = auxi.ranking_metrics(gk_df, gk_ranking, ranking)
	
	wandb.log({"Optimal Cluster": opt_perc, "Mean Average Precision": mAP}) 

	data = [[rho, pval]]
	table = wandb.Table(data=data, columns = ["rho", "pval"])
	wandb.log({"Spearman Correlation" : wandb.plot.scatter(table,"rho", "pval")})

	data = ranking.values.tolist()
	table = wandb.Table(data=data, columns = ['gk_name', 'correct_cluster', 'shots_faced'])
	wandb.log({"GK Ranking": table, "GK bars": wandb.plot.bar(table, "gk_name", "correct_cluster", title="GK Optimal Cluster")})

	return

# Penalty Analysis

def PenaltyAnalysis(args, start='>'):

	'''
		Factors: view_invariant2, penalty_location, hand, height
		Metrics:  Acuracy, Precision, Recall, F1-Score
	'''

	# Pose data
	joined_pose_3d_df, pose_arr, joined_pose_2d_df, pose_2d_arr = auxi.PenaltyImportPose()

	# Percentage of pens that were saved in our dataset
	print(start, "Percentage of saved penalties:", np.mean(joined_pose_3d_df['outcome'] == 'Saved') * 100)

	if args.show or args.save:
		# Show image, image with 2D pose overlay, and 3D pose estimate
		plots.plot_pose_estimation(joined_pose_3d_df, pose_arr, pose_2d_arr, photo_id=315, show=args.show)
		
		pic_ids, path = [388, 20, 3, 243, 302, 377], 'images/pen_images/combined_data/'
		plots.plot_penalty_examples(pose_arr, joined_pose_3d_df, pic_ids, path, show=args.show)
		


	# Get camera-view invariant dataset of 3d poses
	pen_pose_vi = gk.cameraInvariantDataset(pose_arr, vi=args.view_invariant2)
	# Rotates the poses from images taken from behind by 180 degrees
	pen_pose_vi = gk.flipBehindPoses(pen_pose_vi)
	
	# Good Poses DataFrame
	good_poses_3d_df = gk.cleanPenPredictions(joined_pose_3d_df)
	
	# Good Poses Matrix
	last = '47' if joined_pose_3d_df.shape[1] > 40 else '31'
	good_poses_3d_arr = good_poses_3d_df.loc[:,'0':last].values

	# Convert all the good poses to the features space
	poses_features = gk.PenFeatureSpace(good_poses_3d_arr)

	# Fit K-Means model
	kmeans_pens = KMeans(n_clusters=2, random_state = 13).fit(poses_features)
	kmeans_pens_preds = kmeans_pens.predict(poses_features)
	auxi.print_cluster_sizes(kmeans_pens_preds, ['Cluster 0', 'Cluster 1'], start)
	auxi.print_penalty_angles(poses_features, kmeans_pens_preds, start)


	# TSNE representation of body pose
	pens_tsne = TSNE(n_components=2, random_state=29).fit_transform(poses_features)
	
	if args.show:
		plots.plotTSNE(pens_tsne, kmeans_pens_preds, ['Cluster 0', 'Cluster 1'], number=2)

		# GMM - 3D pose, 2D pose viz cluster examples
		ax_array, path = [1, 5, 9, 13, 17], 'images/pen_images/combined_data/'
		plots.penalty_clusterExamples(good_poses_3d_arr, good_poses_3d_df, kmeans_pens_preds, ax_array, path, show=args.show)

	# Save % for clusters
	print()
	auxi.print_save_percentage_cluster(good_poses_3d_df, kmeans_pens_preds, start)

	# Create dataframe of good poses features
	good_poses_feat_df, continuous_var, formula = auxi.createPenaltyGoodPosesFeatures(poses_features, good_poses_3d_df, args)

	# Create dummies for categorical features
	good_poses_feat_df = auxi.PenaltyDummies(good_poses_feat_df, continuous_var, drop_top=True, dummies=True)
	print(start, "New Percentage of saved penalties:", np.mean(good_poses_feat_df['outcome'] == 1) * 100)
	
	# Train/Test Split (70/30)
	df = good_poses_feat_df.copy()
	train_df = df.sample(frac=0.7,random_state=200) #random state is a seed value
	test_df = df.drop(train_df.index)

	X_train, y_train = train_df[train_df.columns[1:]], train_df['outcome']

	# Resampling - because of class imbalance
	undersample = NearMiss(0.4, version=3, n_neighbors=3)
	X_train, y_train = undersample.fit_resample(X_train, y_train)

	#Standardise continuous variables
	scaler = StandardScaler()
	scaler.fit(X_train[continuous_var])
	X_train[continuous_var] = scaler.transform(X_train[continuous_var])
	test_df[continuous_var] = scaler.transform(test_df[continuous_var])
	
	X_test, y_test = test_df[test_df.columns[1:]].copy(), test_df['outcome']

	# Add intercept term
	X_train = sm.add_constant(X_train)
	X_test = sm.add_constant(X_test)

	# Train logistic regression
	log_reg = sm.Logit(y_train, X_train).fit(method='bfgs', maxiter=100)	
	# Logistic regression summary
	print(log_reg.summary())

	sse, ssr, sst, rsquared, f_95, f_value, max_confidence, p_values, cov_matrix = auxi.regression_info(log_reg, X_train, y_train, start)
	# Predictions
	y_pred = log_reg.predict(X_test)
	# Prediction stats
	auxi.printPredictionStats(y_pred, test_df, start)


	# summary = log_reg.summary2()
	if not args.debug:
		TSNE_df = auxi.createTSNEdf(pens_tsne, kmeans_pens_preds, {0:'Cluster 0', 1:'Cluster 1'})
		
		# Log model info
		results_d = {"SSE":sse, "SSR":ssr, "SST":sst, "R-squared":rsquared}
		wandb.log(results_d)
		results_f = {"F-Table[95%]": f_95, "F-Value": f_value, "Max Conf Int": max_confidence}
		wandb.log(results_f)
		wandb.log(p_values)
		table = wandb.Table(data=cov_matrix.values.tolist(), columns=cov_matrix.columns.tolist())
		wandb.log({"Cov Matrix": table})


		wandbPenaltyAnalysis(y_pred, test_df, log_reg, TSNE_df)

	return

def wandbPenaltyAnalysis(y_pred, test_df, model, TSNE_df):
	
	def TSNE_plot():
		data = TSNE_df.values.tolist()
		table = wandb.Table(data=data, columns = ['t-SNE_1', "t-SNE_2", "cluster"])
		wandb.log({"TSNE Penalty" : wandb.plot.scatter(table, 't-SNE_1', 't-SNE_2')})

	TSNE_plot()
	
	
	# Log summary as table
	summary = model.summary2()

	tabela = summary.tables[0]
	df = pd.DataFrame.from_dict({"stats": range(2*len(tabela)), "results": range(2*len(tabela))})
	df['stats'] = pd.concat([tabela[tabela.columns[0]], tabela[tabela.columns[2]]], ignore_index=True)
	df['results']= pd.concat([tabela[tabela.columns[1]], tabela[tabela.columns[3]]], ignore_index=True)
	data = df.values.tolist()
	table = wandb.Table(data=data, columns=df.columns.tolist())
	wandb.log({"Model Summary 0": table})

	df = summary.tables[1].reset_index()
	data = df.values.tolist()
	table = wandb.Table(data=data, columns=df.columns.tolist())
	wandb.log({"Model Summary 1": table})

	
	# Predictions
	y_pred[y_pred < 0.5] = 0
	y_pred[y_pred >= 0.5] = 1


	wandb.log({"conf_mat Penalty" : auxi.plot_cf_matrix(test_df['outcome'].astype(int).tolist(), y_pred.tolist())})
	# cm = wandb.sklearn.plot_confusion_matrix(test_df['outcome'].tolist(), y_pred.astype(int).tolist())
	cmatrix, f1, recall, precision, acc = auxi.classification_metrics(test_df['outcome'].tolist(), np.array(y_pred))
	d = {"Accuracy Penalty": acc, 'F1 Penalty': f1, "Recall Penalty": recall, "Precision Penalty": precision}
	wandb.log(d)

	return

if __name__ == '__main__':
	args = parse_args()
	show_args(args)

	if not args.debug:
		wandb.init(project="LearningFromThePros", entity="luizachagas")
		wandb.config.update(args)


	# print("\t========== 1v1 ANALYSIS ==========\t")
	# # Import and Prepare Data - One on Ones
	# print("* Import and Prepare Data - One on Ones *")
	# set_2d_df, set_3d_df, sets_2d, sets_3d = import_and_prepare()

	# # auxi.create_side_df(sets_2d, set_2d_df, save=True)
	# print("---"*22)

	# # View-Invariance
	# print("* View-Invariance *")
	# sets_3d_cvi_clean, set_3d_cvi_clean_df = viewInvariance(sets_3d, set_3d_df, args)
	# print("---"*22)

	# # Learning Save Technique - Unsupervised Learning
	# print("* Learning Save Technique - Unsupervised Learning *")
	# kmeans_preds = LearningSaveTechnique(sets_3d_cvi_clean, set_3d_cvi_clean_df, args)
	# print("---"*22)

	# # 1v1 Expected Saves Model
	# print("* 1v1 Expected Saves Model *")
	# train_gk_name, test_gk_name, train_df, test_df, svm = ExpectedSavesModel_1v1(set_3d_cvi_clean_df, kmeans_preds, args)
	# print("---"*22)

	# # Pro Goalkeeper Scouting
	# print("* Pro Goalkeeper Scouting *")
	# proGoalkeeperScouting(train_gk_name, test_gk_name , train_df, test_df, svm, args)
	# print("---"*22)

	# Penalty Analysis
	print("\n\n\t========== PENALTY ANALYSIS ==========\t")
	PenaltyAnalysis(args)



	# # Train Linear Regression
	# res = sm.OLS(train_df['outcome'], train_df[train_df.columns[1:]]).fit()
	# print(res.summary())
	# sse, ssr, sst, rsquared = auxi.regression_info(res, train_df['outcome'])
	# # Predictions
	# y_pred = res.predict(test_df[test_df.columns[1:]])
	# # Prediction stats
	# auxi.printPredictionStats(y_pred, test_df)