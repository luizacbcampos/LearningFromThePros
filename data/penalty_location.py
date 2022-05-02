import pandas as pd
import numpy as np


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


def split_url(df_19_21):
	df_19_21['list'] = df_19_21.url.str.split('/')
	df_19_21['match_id'] = df_19_21['list'].str[3]
	df_19_21['date'] = df_19_21['list'].str[4].str.extract(r'(\w+-\d+-\d+)')
	df_19_21['date'] = pd.to_datetime(df_19_21['date'])
	# df_19_21['league'] = df_19_21['list'].str[4].str.extract(r'\w+-\d+-\d+-([\w-]+)')

	return df_19_21.drop(columns='list')

def show_missed(df):
	useful = ['Date', 'Player', 'Team', 'Outcome', 'url', 'pen_taker', 'outcome', 'off_target', 'date']
	print_full(df[df['Outcome'] == 0][useful].sort_values(by=['Player']))
	return

def show_scored(df):
	useful = ['Date', 'Player', 'Team', 'Outcome', 'url', 'pen_taker', 'outcome', 'off_target', 'date']
	print_full(df[df['Outcome'] == 1][useful].sort_values(by=['Player']))
	return

def show_problematic(p_2020):
	duplicate = p_2020.duplicated(subset=['Player','Date', 'Outcome'], keep=False)
	print(p_2020[duplicate])

if __name__ == '__main__':
	
	penalty_df = {}
	penalty_df['2020'] = pd.read_csv("PenaltyKicks/Penalties_2020.csv")
	penalty_df['2020']['Date'] = pd.to_datetime(penalty_df['2020']['Date'])
	penalty_df['2020']['outcome'] = penalty_df['2020']['Outcome'].replace({0: 'Missed', 1: 'Scored'})
	penalty_df['2020'] = penalty_df['2020'].drop_duplicates()
	# print_full(penalty_df['2020'].sort_values(by=['Player', 'Date']))

	# Mais de um penaltis cobrado pelo mesmo jogador numa mesma partica com mesmo outcome: dá problema
	show_problematic(penalty_df['2020'])


	df_19_21 = pd.read_csv("events/prem_pens_19202021.csv")
	df_19_21 = split_url(df_19_21)
	# print(df_19_21)

	df = df_19_21.merge(penalty_df['2020'], how='outer', left_on=['date', 'pen_taker', 'outcome'], right_on=['Date', 'Player', 'outcome'])
	df = df.dropna(subset=['Player'])
	useful = ['Date', 'Player', 'Foot', 'Team', 'Outcome', 'url', 'pen_taker', 'outcome', 'off_target', 'date']
	print(df[useful])

	# show_missed(df)
	# show_scored(df)