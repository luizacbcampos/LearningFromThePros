import numpy
import pandas as pd

def print_full(df, rows=True, columns=False):
	if rows and columns:
		with pd.option_context('display.max_rows', None, 'display.max_columns', None):
			print(df)
	elif rows:
		with pd.option_context('display.max_rows', None):
			print(df)

df = pd.read_csv("events/1v1_events.csv")

# print_full(df['gk_name'].value_counts())

# print(df['gk_name'].unique().tolist())