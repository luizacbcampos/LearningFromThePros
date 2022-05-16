import re
import numpy as np
import pandas as pd
import urllib
import requests
from bs4 import BeautifulSoup

def get_page(URL):
	''' Get page'''
	headers = {'User-Agent': 
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
    #get the page
	pageTree = requests.get(URL, headers=headers)
	#parsing website code to html
	pageSoup = BeautifulSoup(pageTree.content, 'html.parser')
	return pageSoup

def make_link(name):
	''' 
		Search type link. Some examples:
			"https://www.transfermarkt.co.uk/schnellsuche/ergebnis/schnellsuche?query=Neymar&x=0&y=0"
			"https://www.transfermarkt.co.uk/schnellsuche/ergebnis/schnellsuche?query=Lionel+Messi&x=0&y=0"
			"https://www.transfermarkt.co.uk/schnellsuche/ergebnis/schnellsuche?query=Farshad+Noor&x=0&y=0"
	'''
	name = name
	start = "https://www.transfermarkt.co.uk/schnellsuche/ergebnis/schnellsuche?query="
	name = name.replace(" ", "+")
	link = start+name+"&x=0&y=0"
	return link

def search_player(link):
	'''
		Returns if the seach was successfull and the search itself
	'''

	pageSoup = get_page(link)
	try:
		table = pageSoup.find_all('table')[0]
		if table.find('th').get_text() == "Name/Position":
			search = table.find_all('tr', class_=True)
		else:
			raise Exception('Not player table')
	except Exception as e:
		print("Couldn't find player")
		return False, None

	return True, search


def get_link(tr):
	link = None
	try:
		link = tr.find('a', class_="spielprofil_tooltip", href=True)['href']
	except Exception as e:
		try:
			link = tr.find('td', class_="hauptlink").find('a', href=True)['href']
		except Exception as e:
			print("Regular method for link did not work")

	return link


def get_player_feet(link):
	link = "https://www.transfermarkt.co.uk" + link
	pageSoup = get_page(link)
	search = pageSoup.find_all('div', attrs={'class':"box viewport-tracking", 'data-viewport':"Steckbrief"})
	s = search[0].find_all('div', attrs={"class":"spielerdatenundfakten"})
	search = s[0].find('span', class_="info-table__content", text=re.compile("Foot"))
	s = search.find_next('span',class_="info-table__content")
	return s.text
