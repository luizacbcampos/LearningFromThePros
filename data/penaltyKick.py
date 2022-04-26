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
	df = pd.read_csv("PenaltyKicks/penaltyKickTable.csv")
	df['Date'] = pd.to_datetime(df['Date'])
	return df

df = read()

print(df[df['Date'] > pd.to_datetime("2016-12-31")])