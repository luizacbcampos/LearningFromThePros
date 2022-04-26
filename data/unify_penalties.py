import numpy as np
import pandas as pd


def print_full(df, rows=True, columns=False):
	if rows and columns:
		with pd.option_context('display.max_rows', None, 'display.max_columns', None):
			print(df)
	elif rows:
		with pd.option_context('display.max_rows', None):
			print(df)

def read():
	df_17_19 = pd.read_csv("events/prem_pens_17181819.csv")
	df_19_21 = pd.read_csv("events/prem_pens_19202021.csv")
	return df_17_19, df_19_21

def remove_nan_cols(df_17_19, df_19_21):

	# ['Unnamed: 0', 'player_name', 'shot_outcome_name', 'gk_name','Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8','Unnamed: 9', 'Unnamed: 10']
	df_17_19 = df_17_19[['player_name', 'shot_outcome_name', 'gk_name']]

	# ['Unnamed: 0', 'goalkeepers', 'url', 'pen_taker', 'outcome', 'off_target']
	df_19_21 = df_19_21[['goalkeepers', 'url', 'pen_taker', 'outcome', 'off_target']]

	return df_17_19, df_19_21

def add_seasons(df_17_19, df_19_21):
	df_19_21['season'] = '19202021'
	df_17_19['season'] = '17181819'

	return df_17_19, df_19_21

def rename_dfs(df_17_19, df_19_21):
	df_17_19 = df_17_19.rename(columns={'player_name':'pen_taker', 'shot_outcome_name': 'outcome', 'gk_name':'goalkeepers'})
	return df_17_19, df_19_21

def update_off_target(df_17_19, df_19_21):

	df_17_19['off_target'] = np.NaN
	ids = df_17_19[(df_17_19['outcome']=='Off T') | (df_17_19['outcome']=='Post')].index
	df_17_19.at[ids, 'off_target'] = 1
	return df_17_19, df_19_21


def concat_dfs(df_17_19, df_19_21):
	df = pd.concat([df_17_19, df_19_21], ignore_index=True)
	return df

def remove_duplicates(df):
	'''
		Some keepers have more than 1 name version
	'''
	keepers_names = {
	'Alisson Ramsés Becker':'Alisson Becker',
	'Alisson': 'Alisson Becker',
	'David de Gea Quintana': 'David de Gea',
	'Ederson Santana de Moraes': 'Ederson',
	'Kepa Arrizabalaga Revuelta': 'Kepa Arrizabalaga',
	'Rui Pedro dos Santos Patrício': 'Rui Patrício',
	'Vicente Guaita Panadero': 'Vicente Guaita',
	}
	df = df.replace({'goalkeepers': keepers_names})
	return df

df_17_19, df_19_21 = read()

df_17_19, df_19_21 = remove_nan_cols(df_17_19, df_19_21)

df_17_19, df_19_21 = add_seasons(df_17_19, df_19_21)

df_17_19, df_19_21 = rename_dfs(df_17_19, df_19_21)

df_17_19, df_19_21 = update_off_target(df_17_19, df_19_21)

df = concat_dfs(df_17_19, df_19_21)

df = remove_duplicates(df)

# print(df_17_19[['outcome', 'off_target']].value_counts())

# print_full(df['goalkeepers'].value_counts())

# print(df['goalkeepers'].unique().tolist())

df.to_csv("events/prem_pens_all.csv", index=False)