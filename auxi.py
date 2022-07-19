import re
import ast
import cv2
import math
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min, confusion_matrix, f1_score, recall_score, precision_score

from imblearn.under_sampling import NearMiss

from scipy import stats

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mplsoccer import VerticalPitch
import statsmodels.api as sm

import gkpose as gk

direction_dict = {'bottom right corner':1, 'bottom left corner':2, 'centre of the goal':3,
			'top left corner':4, 'top right corner':5, 'top centre of the goal':6,}


def make_train_test(good_poses_feat_df, continuous_var, frac=0.7, perc=0.4):

	# Train/Test Split (70/30)
	df = good_poses_feat_df.copy()
	train_df = df.sample(frac=frac,random_state=200) #random state is a seed value
	test_df = df.drop(train_df.index)

	# Resampling - because of class imbalance
	undersample = NearMiss(perc, version=3, n_neighbors=2)
	X_train, y_train = undersample.fit_resample(train_df[train_df.columns[1:]], train_df['outcome'])

	# re-joing
	train_df = X_train
	train_df["outcome"] = y_train

	#Standardise continuous variables
	scaler = StandardScaler()
	scaler.fit(X_train[continuous_var])
	train_df[continuous_var] = scaler.transform(train_df[continuous_var])
	test_df[continuous_var] = scaler.transform(test_df[continuous_var])

	X_test, y_test = test_df[test_df.columns[1:]].copy(), test_df['outcome']

	test_df = X_test
	test_df['outcome'] = y_test

	# Add intercept term
	train_df = sm.add_constant(train_df)
	test_df = sm.add_constant(test_df)



	return train_df, test_df

def get_t_value(confidence=0.95, degrees_of_freedom=1, side='two-sided'):
    '''
        Retorna o t value dado uma confiança de x% e y graus de liberdade
    '''
    alfa = 1-confidence
    if side=='two-sided':
        return stats.t.ppf(q = 1-alfa/2,df=degrees_of_freedom)
    elif side=='one-sided':
        return stats.t.ppf(q = 1-alfa,df=degrees_of_freedom)
    return -1

def get_z_value(confidence=0.95, side='two-sided'):
    alfa = 1-confidence
    if side=='two-sided':
        return stats.norm.ppf(q = 1-alfa/2)
    elif side=='one-sided':
        return stats.norm.ppf(q = 1-alfa)
    return -1

def get_f_value(confidence=0.95, dfn=1, dfd=1, side='two-sided'):
    '''
        Retorna o f value dado uma confiança de x%, dfn e dfd
    '''
    alfa = 1-confidence
    if side=='two-sided':
        return stats.f.ppf(q = 1-alfa/2, dfn=dfn, dfd=dfd)
    elif side=='one-sided':
        return stats.f.ppf(q = 1-alfa, dfn=dfn, dfd=dfd)
    return -1


def find_duplicates(l):
	l=sorted(l)
	l1, dups = set(), set()
	for i in l:
	    if i not in l1:
	        l1.add(i)
	    else:
	    	dups.add(i)
	return list(dups)


def plot_cf_matrix(y_true, y_pred):

	conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
	fig, ax = plt.subplots(figsize=(7.5, 7.5))
	ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
	for i in range(conf_matrix.shape[0]):
	    for j in range(conf_matrix.shape[1]):
	        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
	 
	plt.xlabel('Predictions', fontsize=18)
	plt.ylabel('Actuals', fontsize=18)
	plt.title('Confusion Matrix', fontsize=18)
	return plt

# metrics

def ranking_metrics(gk_df, gk_ranking, ranking):
	'''
		Get all ranking metrics: accuracy, mAP and Spearman Correlation
	'''
	gt_ranking = ['Neil Leonard Dula Etheridge', 'Łukasz Fabiański', 'Ben Foster', 'Kasper Schmeichel', 
	'David de Gea', 'Hugo Lloris', 'Sergio Rico González', 'Jordan Pickford']

	opt_perc = np.mean(gk_df['chosen_cluster'] == gk_df['optimal_cluster'])

	mAP = mean_average_precision(gt_ranking, ranking.gk_name.tolist())

	rho, pval = SpearmanCorrelation(ranking.gk_name.tolist(), ranking.gk_name.tolist())
	
	return opt_perc, mAP, rho, pval

def classification_metrics(y_true, y_pred):
	cmatrix = confusion_matrix(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	acc = np.mean(y_pred == np.array(y_true))
	return cmatrix, f1, recall, precision, acc

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(y_true, y_pred):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    rs = []
    for t, p in zip(y_true, y_pred):
    	n = 0 if t!=p else 1
    	rs.append(n)
    rs = [rs]
    return np.mean([average_precision(r) for r in rs])

def SpearmanCorrelation(y_true, y_pred):
	'''
		correlation
		Spearman correlation matrix or correlation coefficient 
		(if only 2 variables are given as parameters. Correlation matrix
		is square with length equal to total number of variables (columns or rows) 
		in a and b combined.

		pvalue
		The p-value for a hypothesis test whose null hypotheisis is that two sets of data are uncorrelated. 
		See alternative above for alternative hypotheses. pvalue has the same shape as correlation.
	'''

	rho, pval = stats.spearmanr(y_true, y_pred)
	
	return rho, pval

def get_auto_ftest(res):
	# Teste F
	A = np.identity(len(res.params))[1:,:]
	f_t = res.f_test(A)
	f_value, max_confidence = f_t.fvalue, 1-f_t.pvalue
	f_95 = get_f_value(confidence=0.95, dfn=f_t.df_num, dfd=f_t.df_denom, side='one-sided')
	print("F-Table[95%]: {}; F-Value: {}, Max Conf Int: {}".format(f_95, f_value, max_confidence))
	return f_95, f_value, max_confidence

def regression_info(res, train, outcome, start='>'):


	predicted = res.predict(train)

	sse = np.sum((predicted - outcome)**2)
	# sse = res.ssr
	print(start, "SSE:", sse)

	ssr = np.sum((predicted - outcome.mean())**2)
	# ssr = res.ess
	print(start, "SSR:", ssr)

	sst = ssr + sse
	print(start, "SST:", sst)

	rsquared = 1 - sse/sst
	print(start, "R^2:", rsquared)

	try:
		# Cálculo automático F-teste
		f_95, f_value, max_confidence = get_auto_ftest(res)
	except Exception as e:
		dof_sse = res.df_resid
		dof_ssr = res.df_model

		msr = ssr/dof_ssr
		mse = sse/dof_sse

		f_value = msr/mse
		f_95 = get_f_value(0.95, dfn=dof_ssr, dfd=dof_sse, side='one-sided')
		max_confidence = stats.f.cdf(f_value, dfn=dof_ssr, dfd=dof_sse)
		print("F-Table[95%]: {}; F-Value: {}, Max Conf Int: {}".format(f_95, f_value, max_confidence))


	p_values = res.pvalues.to_dict()
	print("P-values:", p_values)		

	# try:
	# 	cov_matrix = res.cov_params()
	# 	print(cov_matrix)
	# except Exception as e:
	# 	cov_matrix = train.corr()
	# 	print(cov_matrix)

	cov_matrix = train.corr()
	print(cov_matrix)

	['aic', 'bic', 'bse', 'conf_int', 'cov_kwds', 'cov_params', 'cov_type', 'df_model', 'df_resid', 
	'f_test', 'fittedvalues', 'get_margeff', 'initialize', 'k_constant', 'llf', 'llnull', 'llr', 'llr_pvalue', 
	'load', 'mle_retvals', 'mle_settings', 'model', 'nobs', 'normalized_cov_params', 'params', 'pred_table', 
	'predict', 'prsquared', 'pvalues', 'remove_data', 'resid_dev', 'resid_generalized', 'resid_pearson', 
	'resid_response', 'save', 'scale', 'set_null_options', 'summary', 'summary2', 't_test', 't_test_pairwise', 
	'tvalues', 'use_t', 'wald_test', 'wald_test_terms']

	return sse, ssr, sst, rsquared, f_95, f_value, max_confidence, p_values, cov_matrix

# aux

def ImageID(df, array_id):
    '''
        Get photo id's of poses
    '''
    return df.loc[array_id, 'file']

def getClusterNames(number_cluster):

	if number_cluster == 4:
		cluster_name = ['Aggressive Set', 'Passive Set', 'Spread', 'Smother']
	elif number_cluster == 6:
		cluster_name = ['Aggressive Set', 'Passive Set', 'Spread Right', 'Smother Right', 
		'Aggressive Set', 'Passive Set', 'Spread Left', 'Smother Left']
	elif number_cluster == 8:
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

def print_cluster_sizes(kmeans_preds, replace_dict, start='\t>', end='\n'):
	'''
		Print cluster sizes
	'''
	print(start, "Cluster sizes:", end=end)
	d = getClusterSizes(kmeans_preds)

	print('\t'+start, end=' ')
	new_d = {replace_dict[k]:v for k,v in d.items()}
	for k in sorted(new_d.keys()):
		print("{}: {}".format(k, new_d[k]), end='\t')
	print()
	return new_d

def print_cluster_center(set_3d_cvi_clean_df, closest, replace_dict, start='\t>'):
	'''
		Print image that is cluster center
	'''
	print(start, "Closest to cluster center:")
	new_d = {replace_dict[i]:ImageID(set_3d_cvi_clean_df, closest[i]) for i in range(len(closest))}

	print('\t'+start, end=' ')
	for k in sorted(new_d.keys()):
		print("{}: {}".format(k, new_d[k]), end='\t')
	print()
	return new_d

def print_ranking_metrics(gk_df, gk_ranking, ranking, start='\t>'):

	opt_perc, mAP, rho, pval = ranking_metrics(gk_df, gk_ranking, ranking)
	print(start, "Optimal Cluster:", opt_perc)
	print(start, "Mean Average Precision:", mAP)
	print(start, "Spearman Correlation - rho:", rho, "pval:", pval)
	return

def print_classification_metrics(y_test, y_pred, start='\t>'):
	'''
		Easily display classification metrics
	'''
	cmatrix, f1, recall, precision, test_set_acc = classification_metrics(y_test, y_pred)
	display_start = '\t'+start[:-1]+'-'
	print(start, "Classification Metrics:")
	print(display_start, 'Test Set Accuracy:', test_set_acc)
	print(display_start,"F1:", f1)
	print(display_start,"Recall:", recall)
	print(display_start, "Precision:", precision)
	print(display_start, "Confusion Matrix: [[{}, {}], [{}, {}]]".format(cmatrix[0][0], cmatrix[0][1], cmatrix[1][0], cmatrix[1][1]))
	return

def print_penalty_angles(poses_features, kmeans_pens_preds, start='\t'):
	print(start,"Angles:")
	print('\t'+start, "Torso Angle, cluster 0", np.mean(poses_features[kmeans_pens_preds == 0][:,0]))
	print('\t'+start, "Body Angle, cluster 0", np.mean(poses_features[kmeans_pens_preds == 0][:,4]))

	print('\t'+start, "Torso Angle, cluster 1", np.mean(poses_features[kmeans_pens_preds == 1][:,0]))
	print('\t'+start, "Body Angle, cluster 1", np.mean(poses_features[kmeans_pens_preds == 1][:,4]))

def print_save_percentage_cluster(good_poses_3d_df, kmeans_pens_preds, start='>'):
	'''
		Save % for clusters
	'''
	print(start, "Save % for cluster 0 saves:", np.mean(good_poses_3d_df[kmeans_pens_preds == 0]['outcome'] == 'Saved'))
	print(start, "Save % for cluster 1 saves:", np.mean(good_poses_3d_df[kmeans_pens_preds == 1]['outcome'] == 'Saved'))

def printPredictionStats(y_pred, test_df, start='>'):
	'''
		Prints penalty prediction statistics
	'''

	print(start, 'Max xS:', np.max(y_pred))
	print(start, 'Min xS:', np.min(y_pred))
	print(start, 'Mean xS:', np.mean(y_pred))

	y_pred[y_pred < 0.5] = 0
	y_pred[y_pred >= 0.5] = 1

	print_classification_metrics(test_df['outcome'].tolist(), np.array(y_pred), start)
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

# IMPORTS

def importImage(img):
    #Import image
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    return image

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

def PenaltyImportPose():

	#3D pose data
	joined_pose_3d_df, pose_arr = importPenalty3D()
	# 2D pose data
	joined_pose_2d_df, pose_2d_arr = importPenalty2D()
	return joined_pose_3d_df, pose_arr, joined_pose_2d_df, pose_2d_arr


# CREATE DF

def createViewInvariant_df(set_3d_df, sets_3d_cvi):
	'''
		Create the view-invariant dataframe
	'''
	set_3d_cvi_df = pd.DataFrame(sets_3d_cvi)
	set_3d_cvi_df.columns = set_3d_cvi_df.columns.astype(str)
	cols = ['file', 'photo_id', 'under_pressure', 'shot_outcome_name', 'distance_to_goal', 'shot_angle', 'gk_name', 'gk_engage']
	set_3d_cvi_df[cols] = set_3d_df[cols]
	return set_3d_cvi_df

def create_GodPoseFeatDf(poses_features, good_poses_3d_df, keep_indx=False):
	'''
		Create dataframe of good poses features
	'''
	if keep_indx:
		good_poses_feat_df = pd.DataFrame({'outcome': good_poses_3d_df['outcome'], 'indx':good_poses_3d_df['indx']})
		good_poses_feat_df = pd.concat([good_poses_feat_df, pd.DataFrame(poses_features, index=good_poses_feat_df.index)], axis=1)

		# Drops off target strikes 
		good_poses_feat_df = good_poses_feat_df.loc[good_poses_feat_df['outcome']  != 'Off T',:]
		good_poses_feat_df.columns = ['outcome','indx', 'torso_angle','body_height','forward_step', 'hand_height','body_angle']
	
	else:
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
	cluster_dict = {i:i for i in range(len(set(kmeans_preds)))}
	df = make_kmeans_df(kmeans_preds, set_3d_cvi_clean_df, cluster_dict, cluster_names)

	if cluster_names:
		clusters = list(range(len(cluster_names)))
		replace_dict = {k: v for k,v in zip(clusters, cluster_names)}
		df = df.replace({"cluster": replace_dict})

	if save:
		df.to_csv("data/events/cluster_1v1_4.csv", index=False)
	return df

def make_kmeans_df(kmeans_preds, set_3d_cvi_clean_df, cluster_dict, cluster_names=None):

	
	l = [ImageID(set_3d_cvi_clean_df, i) for i in range(len(kmeans_preds))]
	kmeans_dict = {"img_id": l, "cluster": kmeans_preds}
	df = pd.DataFrame.from_dict(kmeans_dict)
	df = df.replace({"cluster": cluster_dict})
	return df

def mean_pose_per_cluster(kmeans_preds, set_3d_cvi_clean_df, cluster_dict, c_center):
	'''
		Creates a df with cluster mean
	'''

	dataframe = set_3d_cvi_clean_df.copy().drop(columns=['photo_id'])
	dataframe['cluster'] = kmeans_preds
	dataframe['cluster'] = dataframe['cluster'].replace(cluster_dict)
	df = dataframe.groupby(by=['cluster']).mean()
	df['cluster_center'] = df.index.map(c_center)
	
	return df.reset_index()


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
		pose = center_pose(pose)
		side = pick_side(pose)
		sides.append(side)
		# print(i, ImageID(set_2d_cvi_clean_df, i), side)

	l = [ImageID(set_2d_cvi_clean_df, i) for i in range(len(sets_2d_cvi_clean))]
	d = {"img_id": l, "side": sides}
	df = pd.DataFrame.from_dict(d).replace({'side': {0: 'Right', 1: "Left"}})
	if save:
		df.to_csv("data/events/side_1v1.csv", index=False)
	return

def PenaltyDummies(good_poses_feat_df, continuous_var, drop_top=True, dummies=True):

	if 'Direction' in good_poses_feat_df.columns:
		# Drop penalties on top zone, trying to fix regression
		if drop_top:
			good_poses_feat_df = good_poses_feat_df[good_poses_feat_df.Direction != 'top right corner']
			good_poses_feat_df = good_poses_feat_df[good_poses_feat_df.Direction != 'top left corner']
			good_poses_feat_df = good_poses_feat_df[good_poses_feat_df.Direction != 'top centre of the goal']
		
		good_poses_feat_df['Direction'] = good_poses_feat_df['Direction'].replace(direction_dict)
		if dummies:
			good_poses_feat_df = pd.get_dummies(good_poses_feat_df, columns=['Direction'])
	if 'Dominant_side' in good_poses_feat_df.columns:
		good_poses_feat_df['Dominant_side'] = good_poses_feat_df['Dominant_side'].replace({'right': 0, 'left': 1})
		if dummies:
			good_poses_feat_df = pd.get_dummies(good_poses_feat_df, columns=['Dominant_side'])

	return good_poses_feat_df.drop(columns=['indx'])

def createPenaltyGoodPosesFeatures(poses_features, good_poses_3d_df, args):
	'''
		Creates the penalty regression dataframe
	'''
	# parser.add_argument('-p', '--penalty_location', type=int, default=0, help='Use penalty location')
	#    parser.add_argument('-ha', '--hand', type=int, default=0, help='Add dominant hand parameter')
	#    parser.add_argument('-he', '--height', type=int, default=0, help='Add goalkeeper height information')

	good_poses_feat_df = create_GodPoseFeatDf(poses_features, good_poses_3d_df, True)
	continuous_var = ['torso_angle','body_height','forward_step','hand_height','body_angle']

	if args.hand or args.height or args.penalty_location:
		tudo = pd.read_csv("data/events/prem_pens_all.csv")
		df = pd.merge(good_poses_feat_df, tudo, left_on='indx', right_on='indx', how='left')
		df = df.drop(columns=['outcome_y', 'off_target', 'pen_taker', 'goalkeepers', 'Foot'])
		df = df.rename(columns={'outcome_x': 'outcome'})

		if not args.hand:
			df = df.drop(columns=['Dominant_side'])
		if not args.height:
			df = df.drop(columns=['Height'])
		if not args.penalty_location:
			df = df.drop(columns=['Direction'])
		good_poses_feat_df = df.copy()

	if args.height:
		continuous_var += ['Height']

	categorical_var = set(good_poses_feat_df.columns) - set(['outcome', 'indx']) - set(continuous_var)
	formula = 'outcome ~ '
	for v in continuous_var:
		formula += v + ' + '
	for v in categorical_var:
		formula += 'C('+ v + ') + '
	formula = formula[:-2] # remove extra +
	return good_poses_feat_df, continuous_var, formula



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

def cluster_correspondence(kmeans_preds, set_3d_cvi_clean_df, cluster_name, start='\t>'):
	'''
		Using the original paper as Ground-Truth find the cluster name from the new cluster results
	'''
	print("{} Cluster Correspondence using {} clusters".format(start, len(cluster_name)))
	df = make_kmeans_df(kmeans_preds, set_3d_cvi_clean_df, cluster_dict = {i:i for i in range(len(cluster_name))})
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
		# print("HERE")
		side_df = pd.read_csv("data/events/side_1v1.csv")
		gt = gt.merge(side_df, on='img_id')
		df = df.merge(side_df, on='img_id')
		
		# join cols
		gt["cluster"] = gt["cluster"] + ' ' + gt["side"]

		g = cluster_proportions_df(df, gt, on=['img_id', 'side'], cols=['cluster_gt', 'cluster_n'])
		# print(g)
		
		perc_gt = cluster_dict(g, by=['perc_gt', 'perc_novo'])
		s_perc_gt = {k:v[0] for k,v in perc_gt.items()}
		# print(s_perc_gt)

		perc_novo = cluster_dict(g, by=['perc_novo', 'perc_gt'])
		s_perc_novo = {k:v[0] for k,v in perc_novo.items()}		
		# print(s_perc_novo)

		# s_perc_gt = remove_left_right(s_perc_gt)
		# s_perc_novo = remove_left_right(s_perc_novo)
		print("Equal??", s_perc_gt==s_perc_novo)


		if set(s_perc_novo.values()) == set(range(8)):
			return dict((v,k) for k,v in s_perc_novo.items())
		elif set(s_perc_gt.values()) == set(range(8)):
			return dict((v,k) for k,v in s_perc_gt.items())
		else:
			missing = list(set(range(8)) - set(s_perc_novo.values()))
			l = find_duplicates(s_perc_novo.values())
			possib = []
			for k,v in s_perc_novo.items():
				if v==l[0]:
					possib.append(k)

			sw = g[g['cluster_n'].isin(l)]
			sw = sw[sw['cluster_gt'].isin(possib)]
			sw = sw[sw.perc_novo == sw.perc_novo.min()]['cluster_gt'].values
			s_perc_novo[sw[0]] = missing[0]
			# print(s_perc_novo)

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

def get_training_test_sets(train_df, test_df, start="\t>"):
	'''
		Get training and testing set X and y
	'''
	# Get training set X and y
	y_train = train_df.pop('shot_outcome_name')
	X_train = train_df.values
	
	# Get test set X and y
	y_test = test_df.pop('shot_outcome_name')
	X_test = test_df.values
	
	print("{} Training Set Size: {}, Test Set Size:{}".format(start, len(X_train), len(X_test)))
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



def center_pose(pose):
	'''
		Center pose so that cervical column's X is on 0
	'''
	
	points = pose.copy()[:, 0].reshape(pose.shape[0])
	
	coluna = points[[6,8,9]]
	col_mean = coluna.mean()
	pose[:, 0] = pose[:, 0] - col_mean
	return pose

if __name__ == '__main__':
	a = 1

	df = pd.read_csv("data/events/1v1_events.csv", index_col=0)
	df = correct_names(df, 'gk_name')
	# print(df)