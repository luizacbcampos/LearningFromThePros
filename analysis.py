import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.api as sm
from statsmodels.stats import weightstats as smw
from statsmodels.stats import descriptivestats as smd
import scipy

from itertools import chain, combinations
import auxi, plots




c = ['Name', 'State', 'Tags', 'Created', 
	'C', 'KMeans Cluster Count', 'grid_search', 'hand', 'height', 'kernel', 'number_dimensions', 
	'penalty_location', 'save', 'show', 'split_sides', 'view_invariant1', 'view_invariant2', 
	'Accuracy 1v1', 'F1 1v1', 'Recall 1v1', 'Precision 1v1', 'Mean Average Precision', 'Optimal Cluster',	
	'Accuracy Penalty',  'F1 Penalty', 'Recall Penalty', 'Precision Penalty', 'R-squared', 'SSE', 'SSR', 'SST'
	]

numerical_1v1 = ['Accuracy 1v1', 'F1 1v1', 'Recall 1v1', 'Precision 1v1', 'Mean Average Precision', 'Optimal Cluster',]
numerical_pen = ['Accuracy Penalty',  'F1 Penalty', 'Recall Penalty', 'Precision Penalty', 'R-squared', 'SSE', 'SSR', 'SST', 'F-Table[95%]', 'F-Value','Max Conf Int']
regression_pen = ['Intercept','body_angle','body_height','forward_step','hand_height','torso_angle','Height','C(Dominant_side)[T.1]','C(Direction)[T.2]','C(Direction)[T.3]']

numerical = numerical_1v1+numerical_pen

args_1v1 = ['view_invariant1', 'number_dimensions', 'split_sides', 'grid_search', 'C', 'kernel']#, 'KMeans Cluster Count']
args_pen = ['view_invariant2', 'penalty_location', 'hand', 'height',]
arguments = args_1v1 + args_pen

# -------------------------

def stdev(x):
	return np.std(x, ddof=1)

def get_t_value(confidence=0.95, degrees_of_freedom=1, side='two-sided'):
    '''
        Retorna o t value dado uma confiança de x% e y graus de liberdade
    '''
    alfa = 1-confidence
    if side=='two-sided':
        return scipy.stats.t.ppf(q = 1-alfa/2,df=degrees_of_freedom)
    elif side=='one-sided':
        return scipy.stats.t.ppf(q = 1-alfa,df=degrees_of_freedom)
    return -1

def get_z_value(confidence=0.95, side='two-sided'):
    alfa = 1-confidence
    if side=='two-sided':
        return scipy.stats.norm.ppf(q = 1-alfa/2)
    elif side=='one-sided':
        return scipy.stats.norm.ppf(q = 1-alfa)
    return -1

def get_f_value(confidence=0.95, dfn=1, dfd=1, side='two-sided'):
    '''
        Retorna o f value dado uma confiança de x%, dfn e dfd
    '''
    alfa = 1-confidence
    if side=='two-sided':
        return scipy.stats.f.ppf(q = 1-alfa/2, dfn=dfn, dfd=dfd)
    elif side=='one-sided':
        return scipy.stats.f.ppf(q = 1-alfa, dfn=dfn, dfd=dfd)
    return -1

def comparando_pareados(a, b, confidence, side='two-sided'):
    ab = a - b
    m, s = ab.mean(), np.std(ab, ddof=1)
    new_s = s/np.sqrt(len(ab))
    if len(ab) < 30:
    	dist_value = get_t_value(confidence, degrees_of_freedom=len(ab)-1, side=side)
    else:
    	dist_value = get_z_value(confidence, side=side)

    ic = [m-s*dist_value, m+s*dist_value]


    zero = not(ic[0] <= 0 and ic[1] >= 0)
    return m, s, ic, zero

# ---------------------------------------

def powerset(iterable, m=1):
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(m,len(s)+1))


def drop_columns(df):
	df = df.drop(columns=['State', 'Tags', 'Created', 'save', 'show',])
	return df

def drop_nans(df):
	g = ['Accuracy 1v1', 'F1 1v1', 'Recall 1v1', 'Precision 1v1', 'Mean Average Precision', 'Optimal Cluster', 
	'Accuracy Penalty',  'F1 Penalty', 'Recall Penalty', 'Precision Penalty', 'R-squared', 'SSE', 'SSR', 'SST']
	df = df.dropna(subset=numerical)
	df = df.dropna(axis=1, how='all') # nan cols
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
	'''
		Makes sure every configuration has the same amount of repetitions
	'''

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
	df_aux = df_aux.drop(eliminate)
	return df_aux

def remove_repetitions(df, min_1v1, min_pen):

	df_1v1 = remove_to_equal_repetitions(df, args_1v1, numerical_1v1, min_1v1)
	df_pen = remove_to_equal_repetitions(df, args_pen, numerical_pen, min_pen)
	return df_1v1, df_pen

def calculate_SS(num, qi):
	ssqi = num * np.power(qi, 2)
	return ssqi

def mean_experimental_errors(df, args, numerical, ground_truth, y_col):
	
	colunas = list(set(df.columns.tolist()) - (set(args +[y_col])))
	exp = df.copy().drop(columns=colunas)
	exp.insert(loc=0, column='I', value=np.ones(len(exp)))

	gt = ground_truth[y_col].values[0] #ground truth value

	# Generate all possible combinations
	comb = [i for i in powerset(args,2)]
	for i, c in enumerate(comb, 5):
		name = ""+ "|".join(c)
		cols = list(c)
		values = 1
		for col in cols:
			values *= exp[col].to_numpy()
		exp.insert(loc=i, column=name, value=values)
		# print(i, c, name, values)


	q = {i: 0 for i in exp.columns.tolist()[:-1]}
	for col in q.keys():
		line = exp[col].to_numpy()
		q[col] = np.dot(line, exp[y_col])/len(line)

	SS = {k: calculate_SS(len(exp), v) for k,v in q.items()}

	SST = sum(SS.values()) - SS['I']
	# print(exp)
	# print(q)
	# print(SST)
	# print(SS)
	return q, SS

def make_factorial(df, args, numerical, t='1v1', name=None):
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

	# auxi.print_full(grouped, True, True)
	if name:
		grouped.to_csv('results/tables/grouped_'+name+'.csv', index=False)
	else:
		grouped.to_csv('results/tables/grouped_'+numerical[0]+'.csv', index=False)

	if t == '1v1':
		ground_truth = grouped[(grouped.view_invariant1 == 1) & (grouped.number_dimensions == -1) & (grouped.split_sides == -1) & (grouped.grid_search == -1)]
	else:
		ground_truth = grouped[(grouped.view_invariant2 == -1) & (grouped.penalty_location == 1) & (grouped.height == 1) & (grouped.hand == 1)]


	exp_results = {i: {'q': 0, 'ss': 0} for i in numerical}
	for num in numerical:
		q, ss = mean_experimental_errors(grouped, args, numerical, ground_truth, y_col=num+'_mean')
		exp_results[num]['q'] = q
		exp_results[num]['ss'] = ss

	# for k, v in exp_results.items():
	# 	print(k)
	# 	print('q: ', v['q'])
	# 	print('ss:', v['ss'])

	exp = pd.DataFrame.from_dict(exp_results, orient='index')
	print(exp)
	q = pd.json_normalize(data=exp['q'])
	q.columns = ['q.'+ i for i in q.columns.tolist()]
	q.index = exp.index
	ss = pd.json_normalize(data=exp['ss'])
	ss.columns = ['ss.'+ i for i in ss.columns.tolist()]
	ss.index = exp.index

	exp = pd.concat([exp, q, ss], axis=1).drop(columns=['q', 'ss'])

	if name:
		exp.to_csv('results/tables/exp_'+name+'.csv', index=False)	
	else:
		exp.to_csv('results/tables/exp_'+numerical[0]+'.csv', index=False)	

	return grouped


def run_stats(df, numerical, name=None):

	d1 = sm.stats.DescrStatsW(df[numerical], ddof=1)
	statistics = pd.DataFrame.from_dict({'name':numerical, 'count': [len(df) for i in range(len(numerical))]})
	statistics['media'], statistics['stdev'] = d1.mean.tolist(), d1.std.tolist()
	statistics['conf_int_95_low'], statistics['conf_int_95_high'] = 0., 0.

	for ind, row in statistics.iterrows():
		m_, s_ = row.media, row.stdev
		count = row.count
		cf = smw._zconfint_generic(m_, s_, 0.05, 'two-sided')
		statistics.at[ind, 'conf_int_95_low'] = cf[0]
		statistics.at[ind, 'conf_int_95_high'] = cf[1]

	statistics['conf_int_95'] = list(zip(statistics['conf_int_95_low'], statistics['conf_int_95_high']))
	statistics = statistics.drop(columns=['conf_int_95_low', 'conf_int_95_high'])
	print(statistics)
	if name:
		statistics.to_csv('results/tables/stats_'+name+'.csv', index=False)
	else:
		statistics.to_csv('results/tables/stats_'+numerical[0]+'.csv', index=False)
	return statistics

def MSE(value, media):
	count = np.prod(value.shape)
	res = np.power((value - media), 2)
	res = (1/(count))*res.sum()
	return res

if __name__ == '__main__':
	# # print(dir(sm.stats))

	# df  = pd.read_csv("data/events/goalkeepers.csv")
	# print(df.Height.std())
	# print(len(df.Name.value_counts()))
	# # print(df.Dominant_side.value_counts())
	# print(df.columns)

	# print(df.Height.mean(), df.Height.min(), df.Height.max())
	# # sns.histplot(data=df, x="Height", color='#a00505ff')
	# # plt.xlabel("Height in cm")
	# # plt.title("Height distribution")
	# # plt.show()
	# exit()


	# ---------------------------------------------------------------
	file = "wandb_export_2022-07-16T11_42_10.048-03_00.csv"
	oneVone = 0

	if oneVone:
		df = pd.read_csv(file)
		
		df = drop_columns(df)
		print("Shape:", df.shape)
		
		df = drop_nans(df)
		print("Shape:", df.shape)
		
		min_1v1, min_pen = repetitions_by_configuration(df)	
		print("Repetitions: 1v1: {}; Pen: {}".format(min_1v1, min_pen))

		df_1v1, df_pen = remove_repetitions(df, min_1v1, min_pen)

		df_1v1 = df_1v1.replace({'view_invariant1':{0: -1}, 'number_dimensions':{3: 1, 2:-1}, 'split_sides':{0: -1}, 'grid_search':{1:-1, 0: 1}})
		df_pen = df_pen.replace({'view_invariant2': {1: -1, 0: 1}, 'penalty_location': {1: -1, 0: 1}, 'hand': {1: -1, 0: 1}, 'height': {1: -1, 0: 1}})
		print('---'*23)

		print("Stats")
		
		print("-- stats 1v1 --")
		run_stats(df_1v1, numerical_1v1)
		make_factorial(df_1v1, args_1v1[:4], numerical_1v1)

		print("-- stats pen --")

		run_stats(df_pen, numerical_pen)
		make_factorial(df_pen, args_pen, numerical_pen, 'p')

	mapp = 0
	if mapp:
		print("--- Map analysis ---")
		
		df = pd.DataFrame([], columns=args_1v1[:4]+['file', 'up'])
		
		for name in os.listdir('./results/maps/'):
			up = False
			if name[len(name)-4:]=='.npy':
				names = name.split('xs_map_')[1]
				if names[:2] == 'up':
					up = True
				names = names.split('up_')[-1]
				names = names.split('-')
				d = {'up': [int(up)], 'view_invariant1': [int(names[0][-1])], 'number_dimensions': [int(names[1][-1])], 
				'grid_search': [int(names[2][-1])],'split_sides': [int(names[3][-1])], 'file': [name]}
				df2 = pd.DataFrame.from_dict(d)
				df = pd.concat([df, df2])

		df = df.sort_values(['view_invariant1', 'number_dimensions', 'split_sides', 'grid_search', 'up'])
		df = df.drop_duplicates(subset=['view_invariant1', 'number_dimensions', 'split_sides', 'grid_search', 'up'])
		df = df.reset_index(drop=True)
		df = df.replace({'view_invariant1':{0: -1}, 'number_dimensions':{3: 1, 2:-1}, 'split_sides':{0: -1}, 'grid_search':{1:-1, 0: 1}})
		# print(df)

		xs_map, xs_map_up = [], []
		

		for i, row in df.iterrows():
			file = row['file']
			if row['up']:
				xs_map_up.append(np.load('results/maps/'+file))
			else:
				xs_map.append(np.load('results/maps/'+file))
		

		xs_map, xs_map_up = np.array(xs_map), np.array(xs_map_up)

		gt_xs_map = df[(df.view_invariant1 == 1) & (df.number_dimensions == -1) & (df.split_sides == -1) & (df.grid_search == -1) & (df.up == 0)]['file'].values[0]
		gt_xs_map = np.load('results/maps/'+ gt_xs_map)
		gt_xs_map_up = df[(df.view_invariant1 == 1) & (df.number_dimensions == -1) & (df.split_sides == -1) & (df.grid_search == -1) & (df.up == 1)]['file'].values[0]
		gt_xs_map_up = np.load('results/maps/'+ gt_xs_map_up)


		mean_xs_map = xs_map.mean(axis=0)
		mean_xs_map_up = xs_map_up.mean(axis=0)
		cluster_name = ['Aggressive Set', 'Passive Set', 'Spread', 'Smother']
		plots.plotBestTechniqueUp(mean_xs_map, mean_xs_map_up, cluster_name, show=True, title='Mean xSAA map')
		
		df['mse'] = 0.
		mse, mse_up = [], []

		for xs in xs_map:
			mse.append(MSE(xs, mean_xs_map))

		for xs in xs_map_up:
			mse_up.append(MSE(xs, mean_xs_map_up))

		df.loc[df['up'] == 0, 'mse'] = mse
		df.loc[df['up'] == 1, 'mse'] = mse_up

		df['mse_1'] = [mse[int(i/2)] for i in range(2*len(mse))]
		df['mse_up'] = [mse_up[int(i/2)] for i in range(2*len(mse_up))]
		
		# print(df)
		# run_stats(df, ['mse', 'mse_1', 'mse_up'], 'map')
		# make_factorial(df, args_1v1[:4], ['mse', 'mse_1', 'mse_up'], '1v1', name='map')

		# for i, row in df.iterrows():
		# 	print(int(i/2))
		# 	name = 'v1{}-nd{}-g{}-si{}'.format(row['view_invariant1'], row['number_dimensions'], row['grid_search'], row['split_sides'])
		# 	plots.plotBestTechniqueUp(xs_map[int(i/2)], xs_map_up[int(i/2)], cluster_name, show=False, title=name+' xSAA map').savefig('figures/'+name+'.png')


		# print(df[args_1v1[:4]].drop_duplicates().to_numpy().reshape(4,4,4))

		# print(df)

	gk=0
	if gk:
		print("--- GK Analysis ---")
		df = pd.read_csv("gk_table.csv")
		df = df.replace({'view_invariant1':{0: -1}, 'number_dimensions':{3: 1, 2:-1}, 'split_sides':{0: -1}, 'grid_search':{1:-1, 0: 1}})
		df = df.drop_duplicates()

		# print(df)
		print("Stats")

		shots_faced = {g: n for g, n in zip(df['gk_name'].tolist(), df['shots_faced'].tolist())}

		cols = args_1v1[:4]
		for gk in df['gk_name'].unique().tolist():
			name = 'cc.'+gk
			cols.append(name)

		novo = pd.DataFrame([], columns=cols)

		run_stats(df, ['correct_cluster', 'shots_faced'])

		grouped = df.groupby(args_1v1[:4])
		for k, v in grouped:
			g, si,v1, nd = k
			d = {'view_invariant1':[v1], 'number_dimensions':[nd], 'split_sides':[si], 'grid_search':[g]}
			for i, row in v.iterrows():
				name = 'cc.'+row['gk_name']
				d[name] = [row['correct_cluster']]

			df2 = pd.DataFrame.from_dict(d)
			novo = pd.concat([novo, df2])

		numerical_gk = list(set(cols) - set(args_1v1[:4]))
		novo = novo.reset_index(drop=True)
		print(novo)


		s = run_stats(novo, numerical_gk, name='gk')
		make_factorial(novo, args_1v1[:4], numerical_gk, '1v1', name='gk')


		s['CI_95_0'], s['CI_95_1'] = s['conf_int_95'].str
		s = s.drop(columns=['conf_int_95'])

		comb = [c for c in combinations(numerical_gk, 2)]

		for c in comb:
			print(c)
			one, two = c
			values_one = novo[one].to_numpy()
			values_two = novo[two].to_numpy()
			m, s, ic, zero = comparando_pareados(values_one, values_two, 0.95, side='two-sided')
			print("\tmean: {}; std: {}, IC:{}, Diff: {}".format(m, s, ic, zero))

	pen_model = 1
	if pen_model:
		print("--- Pen Model Analysis ---")
		df = pd.read_csv("pen_model_table.csv")
		df = df.replace({'view_invariant2': {1: -1, 0: 1}, 'penalty_location': {1: -1, 0: 1}, 'hand': {1: -1, 0: 1}, 'height': {1: -1, 0: 1}})
		df = df.drop_duplicates()

		df['variance'] = np.power(df['Std.Err.'], 2)
		grouped = df.groupby(['variable'])
		for k, v in grouped:
			print("{}, count = {}".format(k, len(v)))
			media = v['Coef.'].mean()
			media_var = v['variance'].mean()
			std_error = np.sqrt(media_var)
			t_value = get_t_value(confidence=0.95, degrees_of_freedom=len(v)-1, side='two-sided')
			ic = [media - t_value*std_error, media + t_value*std_error]
			# print(v.columns)
			print(media, std_error, ic)

			# print(k,v.mean())
			# exit()

		# run_stats(df, ['Coef.','Std.Err.','z','P>|z|','[0.025','0.975]'], 'delete')
		# print(df)

