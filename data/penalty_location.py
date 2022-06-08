import numpy as np
import pandas as pd

import warnings

# --- AUX ---
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

def player_name_dict():
	d = {
		"Aleksandar Mitrovic": "Aleksandar Mitrović",
		"Alexis Alejandro Sánchez Sánchez": "Alexis Sánchez",
		"Alisson Ramsés Becker": "Alisson",
		"André Ayew": "André Ayew Pelé",
		"Christian Benteke Liolo": "Christian Benteke",
		"Daniel William John Ings": "Danny Ings",
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

# Load
def read_penalties(path):
	pen_df = pd.read_csv(path)
	pen_df['Date'] = pd.to_datetime(pen_df['Date'])
	pen_df['outcome'] = pen_df['Outcome'].replace({0: 'Missed', 1: 'Scored'})
	pen_df['Player'] = pen_df['Player'].replace(player_name_dict())
	return pen_df.drop_duplicates()

def read():
	df_17_19 = pd.read_csv("events/prem_pens_17181819.csv", index_col=0)
	df_17_19['player_name'] = df_17_19['player_name'].replace(player_name_dict())
	df_17_19['gk_name'] = df_17_19['gk_name'].replace(player_name_dict())

	df_19_21 = pd.read_csv("events/prem_pens_19202021.csv", index_col=0)
	df_19_21['pen_taker'] = df_19_21['pen_taker'].replace(player_name_dict())
	df_19_21['goalkeepers'] = df_19_21['goalkeepers'].replace(player_name_dict())
	return df_17_19, df_19_21

def load_penalties():
	penalty_df = {}
	penalty_df['2017'] = read_penalties("PenaltyKicks/Penalties_2017.csv")
	penalty_df['2018'] = read_penalties("PenaltyKicks/Penalties_2018.csv")
	penalty_df['2019'] = read_penalties("PenaltyKicks/Penalties_2019.csv")
	penalty_df['2020'] = read_penalties("PenaltyKicks/Penalties_2020.csv")
	penalty_df['2021'] = read_penalties("PenaltyKicks/Penalties_2021.csv")
	return penalty_df


# Edit df

def concat_dfs(penalty_df):
	'''
		Concatenate all the years scrapped
	'''
	pen_df = pd.concat(penalty_df.values(), axis=0, ignore_index=True)
	return pen_df.sort_values(by=['Date', 'Player'])

def split_url(df_19_21):
	df_19_21['list'] = df_19_21.url.str.split('/')
	df_19_21['match_id'] = df_19_21['list'].str[3]
	df_19_21['date'] = df_19_21['list'].str[4].str.extract(r'(\w+-\d+-\d+)')
	df_19_21['date'] = pd.to_datetime(df_19_21['date'])
	# df_19_21['league'] = df_19_21['list'].str[4].str.extract(r'\w+-\d+-\d+-([\w-]+)')

	return df_19_21.drop(columns='list')

def edit_2017(df_17_19):

	df_17_19['date'] = pd.to_datetime(df_17_19['date'], format='%d-%m-%Y')
	
	#Change shots that hit post to 'Off T'
	df_17_19.loc[df_17_19['shot_outcome_name'] == 'Post', 'shot_outcome_name'] = 'Off T'
	
	df_17_19 = df_17_19.rename(columns={'player_name': 'pen_taker', 'gk_name':'goalkeepers', "shot_outcome_name": "outcome"})
	df_17_19['off_target'] = np.NaN

	return df_17_19

def cleanPenDataFrames(df_19_21, df_17_19):
    #df_17_19: 17 - 19 data
    #df_19_21: 19 - 21 data

    def clean_2017_2019(df_17_19):
    	df_17_19.loc[df_17_19['outcome'] == 'Goal', 'outcome'] = 'Scored'
    	df_17_19.loc[df_17_19['outcome'] == 'Saved', 'outcome'] = 'Missed'

    	df_17_19.loc[df_17_19['outcome'] == 'Off T', 'off_target'] = 1
    	df_17_19.loc[df_17_19['outcome'] == 'Off T', 'outcome'] = 'Missed'
    	
    	df_17_19 = df_17_19.drop(columns=[ 'player_team', 'gk_team', 'season']).dropna(axis='columns', how='all')
    	df_17_19[['pen_taker','outcome', 'off_target', 'goalkeepers', 'date']]
    	
    	return df_17_19

    def clean_2019_2021(df_19_21):
    	df_19_21.drop(columns=['url', 'match_id'], inplace=True)
    	df_19_21 = df_19_21[['pen_taker','outcome','off_target', 'goalkeepers', 'date']]
    	return df_19_21
    
    df_17_19 = clean_2017_2019(df_17_19)
    df_19_21 = clean_2019_2021(df_19_21)
    
    joined_df = df_17_19.append(df_19_21, ignore_index=True) #contains all pens
    return joined_df



def Player_Names(penalty_df, df_joined):
	
	print("Player names: ")
	# pen_taker, outcome, goalkeepers, date
	player_set = set(df_joined['pen_taker'].unique().tolist())

	for k in penalty_df.keys():
		year_list = []
		player_list = penalty_df[k]['Player'].unique().tolist()
		for player in player_list:
			if player not in player_set:
				year_list.append(player)
		
		if len(year_list) > 0:
			print("{}:\t{}".format(k, year_list))
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
	print_full(df[duplicate])
	print("---"*20)

def merge_dfs(pen_df, joined_df):

	df = joined_df.merge(pen_df, how='left', left_on=['date', 'pen_taker', 'outcome'], right_on=['Date', 'Player', 'outcome'])
	return df

def missing_info_17_19(df):
	
	df['name'] = ''
	df = df[(df['gk_team']=='3') | (df['player_team']=='team') | (df['season'].isnull()) | (df['date'].isnull())]
	name_count = pd.concat([df['pen_taker'], df['goalkeepers']])
	name_count = name_count.value_counts().reset_index(level=0).rename({'index':'name', 0: 'count'}, axis = 'columns')
	
	for index, row in df.iterrows():
		gk, player = row['goalkeepers'], row['pen_taker']
		gk_c, player_c = name_count[name_count['name'] == gk]['count'].values[0], name_count[name_count['name'] == player]['count'].values[0]

		if player_c > gk_c:
			df.at[index, 'name'] = player
		else:
			df.at[index, 'name'] = gk
	df = df.sort_values(['name', 'goalkeepers', 'pen_taker'])
	return df.drop(columns='name')

def goalkeeper_list(df):
	keepers_names = df['goalkeepers'].unique().tolist()

	return sorted(keepers_names)

def get_keepers_info():
	
	from PenaltyKicks.transfermarkt import get_player_feet
	df = pd.read_csv("events/goalkeepers.csv")
	for index, row in df.iterrows():
		side = get_player_feet(row['Link'])
		df.at[index, 'Dominant_side'] = side
	df.to_csv("events/goalkeepers.csv", index=False)	
	return df

if __name__ == '__main__':
	
	penalty_df = load_penalties()
	# print_full(penalty_df['2020'].sort_values(by=['Player', 'Date']))

	df_17_19, df_19_21 = read()
	# print(df_17_19)
	df_19_21 = split_url(df_19_21)
	df_17_19 = edit_2017(df_17_19)
	
	joined_df = cleanPenDataFrames(df_19_21.copy(), df_17_19.copy())


	# print(df_19_21)
	Player_Names(penalty_df, joined_df)

	pen_df = concat_dfs(penalty_df)
	print(pen_df[['Direction', 'outcome']].value_counts())

	# Mais de um penalti cobrado pelo mesmo jogador numa mesma partica com mesmo outcome: dá problema
	show_problematic(pen_df)

	df = merge_dfs(pen_df, joined_df)
	# df = df.dropna(subset=['Player'])

	useful = ['Date', 'Player', 'Foot', 'Team', 'Outcome', 'pen_taker', 'outcome', 'goalkeepers', 'date']
	dates = show_missing(df[useful])

	for key, value in dates.items():
		print('{}: {}'.format(key, value))

	# print(df[df['date'].isnull()])

	print("///"*25, '\n')


	print("Looking at df_17_19: ")

	
	print("Missing info:")
	print_full(missing_info_17_19(df_17_19.copy()))

	print("erro:")
	print(df_17_19[df_17_19['player_team'] == df_17_19['gk_team']])
	
	# get_keepers_info()
	

	