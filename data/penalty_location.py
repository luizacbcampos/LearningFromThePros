import numpy as np
import pandas as pd

import warnings

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

# Load
def read_penalties(path):
	pen_df = pd.read_csv(path)
	pen_df['Date'] = pd.to_datetime(pen_df['Date'])
	pen_df['outcome'] = pen_df['Outcome'].replace({0: 'Missed', 1: 'Scored'})
	pen_df['Player'] = pen_df['Player'].replace(player_name_dict())
	return pen_df.drop_duplicates()

def read():
	df_17_19 = pd.read_csv("events/prem_pens_17181819.csv")
	df_17_19['player_name'] = df_17_19['player_name'].replace(player_name_dict())
	df_19_21 = pd.read_csv("events/prem_pens_19202021.csv")
	return df_17_19, df_19_21

def load_penalties():
	penalty_df = {}
	penalty_df['2017'] = read_penalties("PenaltyKicks/Penalties_2017.csv")
	penalty_df['2020'] = read_penalties("PenaltyKicks/Penalties_2020.csv")
	penalty_df['2021'] = read_penalties("PenaltyKicks/Penalties_2021.csv")
	return penalty_df

def concat_dfs(penalty_df):
	pen_df = pd.concat(penalty_df.values(), axis=0, ignore_index=True)
	return pen_df.sort_values(by=['Date', 'Player'])

# Edit df

def split_url(df_19_21):
	df_19_21['list'] = df_19_21.url.str.split('/')
	df_19_21['match_id'] = df_19_21['list'].str[3]
	df_19_21['date'] = df_19_21['list'].str[4].str.extract(r'(\w+-\d+-\d+)')
	df_19_21['date'] = pd.to_datetime(df_19_21['date'])
	# df_19_21['league'] = df_19_21['list'].str[4].str.extract(r'\w+-\d+-\d+-([\w-]+)')

	return df_19_21.drop(columns='list')

def player_name_dict():
	d = {
		"Aleksandar Mitrovic": "Aleksandar Mitrović",
		"Alexis Alejandro Sánchez Sánchez": "Alexis Sánchez",
		"Daniel William John Ings": "Danny Ings",
		"Dusan Tadic": "Dušan Tadić",
		"Gabriel Fernando de Jesus": "Gabriel Jesus",
		"Gylfi Sigurdsson": "Gylfi Sigurðsson",
		"Gylfi Þór Sigurðsson": "Gylfi Sigurðsson",
		"Ilkay Gündogan": "İlkay Gündoğan",
		"James Philip Milner": "James Milner",
		"Jorge Luiz Frello Filho": "Jorginho",
		"Joseph Willock": "Joe Willock",
		"Luka Milivojevic": "Luka Milivojević",
		"Marko Arnautovic": "Marko Arnautović",
		"Raheem Shaquille Sterling": "Raheem Sterling",
		"Raúl Alonso Jiménez Rodríguez": "Raúl Jiménez",
		"Rodrigo": "Rodri",
		"Rúben Diogo Da Silva Neves": "Rúben Neves", 
		"Sergio Leonel Agüero del Castillo": "Sergio Agüero",
		"Son Heung-Min": "Son Heung-min",
		"Wayne Mark Rooney": "Wayne Rooney",
		"Willian Borges da Silva": "Willian",
	}
	return d

def Player_Names(penalty_df, df_19_21, df_17_19):
	
	player_set = set(pd.concat([df_19_21['pen_taker'], df_17_19['player_name']]).unique().tolist())

	for k in penalty_df.keys():
		print(k)
		player_list = penalty_df[k]['Player'].unique().tolist()
		for player in player_list:
			if player not in player_set:
				print(player)
		print("---"*20)

	print(sorted(player_set))
	print("___"*20)


def show_missed(df):
	useful = ['Date', 'Player', 'Team', 'Outcome', 'url', 'pen_taker', 'outcome', 'off_target', 'date']
	print_full(df[df['Outcome'] == 0][useful].sort_values(by=['Player']))
	return

def show_scored(df):
	useful = ['Date', 'Player', 'Team', 'Outcome', 'url', 'pen_taker', 'outcome', 'off_target', 'date']
	print_full(df[df['Outcome'] == 1][useful].sort_values(by=['Player']))
	return

def show_missing(df, prt=False):
	print("Missing: ")
	dates = {}

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		df['year'] = df['date'].apply(lambda x: x.year)

	for yr in df['year'].unique().tolist():
		f = df[(df['year'] == yr) & (df['Player'].isnull())]
		if prt:
			print(yr)
			print_full(f)

		dates[yr] = f['date'].dt.strftime('%Y-%m-%d').unique().tolist()

	return dates

def show_problematic(df):
	'''
		Mais de um penaltis cobrado pelo mesmo jogador numa mesma partica com mesmo outcome: dá problema
	'''
	print("Problematic rows: ")
	duplicate = df.duplicated(subset=['Player','Date', 'Outcome'], keep=False)
	print(df[duplicate])
	print("---"*20)



if __name__ == '__main__':
	
	penalty_df = load_penalties()
	# print_full(penalty_df['2020'].sort_values(by=['Player', 'Date']))

	df_17_19, df_19_21 = read()
	# print(df_17_19)
	df_19_21 = split_url(df_19_21)
	# print(df_19_21)
	Player_Names(penalty_df, df_19_21, df_17_19)


	pen_df = concat_dfs(penalty_df)
	# Mais de um penaltis cobrado pelo mesmo jogador numa mesma partica com mesmo outcome: dá problema
	show_problematic(pen_df)

	df = df_19_21.merge(pen_df, how='outer', left_on=['date', 'pen_taker', 'outcome'], right_on=['Date', 'Player', 'outcome'])
	# df = df.dropna(subset=['Player'])
	useful = ['Date', 'Player', 'Foot', 'Team', 'Outcome', 'url', 'pen_taker', 'outcome', 'off_target', 'date']
	dates = show_missing(df[useful])

	for key, value in dates.items():
		print('{}: {}'.format(key, value))

	print(df[df['date'].isnull()])

	# print_full(df_17_19)

	# show_missed(df)
	# show_scored(df)

	print("///"*25)
	dele = pd.read_csv("dele.csv")
	print(dele[dele['gk_team'] == '3'].drop_duplicates(subset=['gk_name']))

	print(dele[(dele['player_team'] == 'team') & (dele['season'].isnull())].drop_duplicates(subset=['player_name']))