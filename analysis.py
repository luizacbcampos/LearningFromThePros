import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.api as sm
from statsmodels.stats import weightstats as smw
from statsmodels.stats import descriptivestats as smd

c = ['Name', 'State', 'Tags', 'Created', 
	'C', 'KMeans Cluster Count', 'grid_search', 'hand', 'height', 'kernel', 'number_dimensions', 
	'penalty_location', 'save', 'show', 'split_sides', 'view_invariant1', 'view_invariant2', 
	'Accuracy 1v1', 'F1 1v1', 'Recall 1v1', 'Precision 1v1', 'Mean Average Precision', 'Optimal Cluster',	
	'Accuracy Penalty',  'F1 Penalty', 'Recall Penalty', 'Precision Penalty', 'R-squared', 'SSE', 'SSR', 'SST'
	]

numerical_1v1 = ['Accuracy 1v1', 'F1 1v1', 'Recall 1v1', 'Precision 1v1', 'Mean Average Precision', 'Optimal Cluster',]
numerical_pen = ['Accuracy Penalty',  'F1 Penalty', 'Recall Penalty', 'Precision Penalty', 'R-squared', 'SSE', 'SSR', 'SST'] 
numerical = numerical_1v1+numerical_pen

args_1v1 = ['view_invariant1', 'number_dimensions', 'split_sides', 'grid_search', 'C', 'kernel']#, 'KMeans Cluster Count']
args_pen = ['view_invariant2', 'penalty_location', 'hand', 'height',]
arguments = args_1v1 + args_pen

def stdev(x):
	return np.std(x, ddof=1)

def drop_columns(df):
	df = df.drop(columns=['State', 'Tags', 'Created', 'save', 'show',])
	return df

def drop_nans(df):
	g = ['Accuracy 1v1', 'F1 1v1', 'Recall 1v1', 'Precision 1v1', 'Mean Average Precision', 'Optimal Cluster', 
	'Accuracy Penalty',  'F1 Penalty', 'Recall Penalty', 'Precision Penalty', 'R-squared', 'SSE', 'SSR', 'SST']
	df = df.dropna(subset=g)
	return df

def repetitions_by_configuration(df):
	df['count'] = 0

	df_1v1 = df.groupby(by=args_1v1).count().reset_index(col_level=1)[args_1v1+['count']]
	# print('1v1')
	# print(df_1v1)

	df_pen = df.groupby(by=args_pen).count().reset_index(col_level=1)[args_pen+['count']]
	# print("Penalty")
	# print(df_pen)
	return df_1v1['count'].min(), df_pen['count'].min()

def remove_to_equal_repetitions(df, args, num, min_aux):
	
	df_aux = df.sort_values(by=args)[args+num]
	df_aux['indx'] = df_aux.index

	grouped = df_aux.groupby(by=args)
	eliminate = []
	for key, values in grouped:
		indexes = values.indx.tolist()
		if len(indexes) > min_aux:
			for i in range(len(indexes)-min_aux):
				eliminate.append(indexes[i])
		# break

	df_aux = df_aux.drop(columns=['indx'])
	return df_aux.drop(eliminate)

def remove_repetitions(df, min_1v1, min_pen):

	df_1v1 = remove_to_equal_repetitions(df, args_1v1, numerical_1v1, min_1v1)
	df_pen = remove_to_equal_repetitions(df, args_pen, numerical_pen, min_pen)
	return df_1v1, df_pen

def make_factorial(df, args, numerical):
	grouped = df[args+numerical].groupby(by=args).agg([np.mean, stdev, list])
	grouped.columns = ["_".join(x) for x in grouped.columns.to_numpy().ravel()]
	grouped = grouped.reset_index(col_level=0)

	for col in numerical:
		grouped[col+'_SSE'] = 0
		grouped[col+'_conf_int_95_low'] = 0.
		grouped[col+'_conf_int_95_high'] = 0.

	for ind, row in grouped.iterrows():
		for col in numerical:
			outcome = np.array(row[col+'_list'])
			sse = np.sum((row[col+'_mean'] - outcome)**2)
			grouped.at[ind, col+'_SSE'] = sse
	
			m_, s_ = row[col+'_mean'], row[col+'_stdev']
			cf = smw._zconfint_generic(m_, s_, 0.05, 'two-sided') 
			grouped.at[ind, col+'_conf_int_95_low'] = cf[0]
			grouped.at[ind, col+'_conf_int_95_high'] = cf[0]
	
	listas = [col+'_list' for col in numerical]
	grouped = grouped.drop(columns=listas)
	for col in numerical:
		grouped[col+'_conf_int_95'] = list(zip(grouped[col+'_conf_int_95_low'], grouped[col+'_conf_int_95_high']))
		grouped = grouped.drop(columns=[col+'_conf_int_95_low', col+'_conf_int_95_high'])

	print(grouped)
	grouped.to_csv('grouped_'+numerical[0]+'.csv', index=False)

	return grouped


def run_stats(df, numerical):
	
	d1 = sm.stats.DescrStatsW(df[numerical], ddof=1)
	statistics = pd.DataFrame.from_dict({'name':numerical, 'count': [len(df) for i in range(len(numerical))]})
	statistics['media'], statistics['stdev'] = d1.mean.tolist(), d1.std.tolist()
	statistics['conf_int_95_low'], statistics['conf_int_95_high'] = 0., 0.

	for ind, row in statistics.iterrows():
		m_, s_ = row.media, row.stdev
		cf = smw._zconfint_generic(m_, s_, 0.05, 'two-sided')
		statistics.at[ind, 'conf_int_95_low'] = cf[0]
		statistics.at[ind, 'conf_int_95_high'] = cf[1]

	statistics['conf_int_95'] = list(zip(statistics['conf_int_95_low'], statistics['conf_int_95_high']))
	statistics = statistics.drop(columns=['conf_int_95_low', 'conf_int_95_high'])
	print(statistics)
	statistics.to_csv('stats_'+numerical[0]+'.csv', index=False)
	return statistics



if __name__ == '__main__':
	# print(dir(sm.stats))

	df  = pd.read_csv("data/events/goalkeepers.csv")
	# print(df.Dominant_side.value_counts())
	print(df.columns)

	print(df.Height.mean(), df.Height.min(), df.Height.max())
	# sns.histplot(data=df, x="Height", color='#a00505ff')
	# plt.xlabel("Height in cm")
	# plt.title("Height distribution")
	# plt.show()
	exit()


	# ---------------------------------------------------------------
	file = "imgs/wandb_export_2022-06-20T14_01_33.971-03_00.csv"
	df = pd.read_csv(file)
	df = drop_columns(df)
	print("Shape:", df.shape)

	df = drop_nans(df)
	print("Shape:", df.shape)
	# print(df.columns)
	
	min_1v1, min_pen = repetitions_by_configuration(df)	

	df_1v1, df_pen = remove_repetitions(df, min_1v1, min_pen)

	df_1v1 = df_1v1.replace({'view_invariant1':{0: -1}, 'number_dimensions':{3: 1, 2:-1}, 'split_sides':{0: -1}, 'grid_search':{1:-1, 0: 1}})
	df_pen = df_pen.replace({'view_invariant2': {1: -1, 0: 1}, 'penalty_location': {1: -1, 0: 1}, 'hand': {1: -1, 0: 1}, 'height': {1: -1, 0: 1}})

	print("Stats")
	
	print("-- stats 1v1 --")
	run_stats(df_1v1, numerical_1v1)
	make_factorial(df_1v1, args_1v1[:4], numerical_1v1)

	# print(smd.describe(df_1v1)) #mais opções
	print("-- stats pen --")

	run_stats(df_pen, numerical_pen)
	make_factorial(df_pen, args_pen, numerical_pen)