import os
from bs4 import BeautifulSoup
from requests_html import HTMLSession

# import urllib2
# import urllib.request

class DateScraper:

    # Initialize a new scraper of the date page on espn.
    def __init__(self, date,timeout=15):
        self.date = date
        # self.datePageUrl = "http://www.espnfc.us/scores?date=" + date
        self.datePageUrl = "https://www.espn.com/soccer/scoreboard?date=" + date 
        self.beautifulSoup = None
        self.allGames = None
        self.session = HTMLSession()
        self.timeout = timeout

    def getDate(self):
        return self.date

    def getDatePageUrl(self):
        return self.getDatePageUrl

    def getBeautifulSoup(self):
        return self.beautifulSoup

    def getAllGames(self):
        return self.allGames

    def getAmountGames(self):
        if self.allGames != None:
            return len(self.allGames)
        return -1
    def getSession(self):
        return self.session
        
    def closeSession(self):
        self.session.close()


    # Make a BeautifulSoup of that date page
    def makeBeautifulSoup(self):
        # self.beautifulSoup = BeautifulSoup(urllib.request.urlopen(self.datePageUrl), 'html5lib')
        # Must use html5lib to correctly scrape the page. Doesn't work without this.

        print(self.datePageUrl)
        headers = {'User-Agent': 
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
        r = self.session.get(self.datePageUrl, headers=headers, timeout=self.timeout)
        r.html.render()
        self.beautifulSoup = BeautifulSoup(r.html.raw_html, "html.parser")



    def makeListOfgames(self):
        listOfGames = []
        # mainBody = self.beautifulSoup.find_all("div", {"id" :["main"]})
        scoreboard = self.beautifulSoup.find("div", {"id" :["scoreboard-page"]})
        mainBody = scoreboard.find("div", {"id" :["events"]})
        mainBody = mainBody.find_all("article",{"class": "scoreboard"})
        # print(len(mainBody))

        # Returns the list of all relevant games.
        for game in mainBody:
            # print(game.prettify())
            individualGames = game.find_all("a", {"class":"button-alt sm", "href": True})


            for individualGame in individualGames:
                this_game = individualGame["href"]
                this_game = this_game.split('gameId')[-1]
                this_game = this_game[1:]
                # print(this_game[1:])
                listOfGames.append(this_game)

        self.allGames = list(set(listOfGames))

    def writeGameList(self,date):
        try:
            path = "./Games/%s/%s/%s" % (str(date.year), str(date.month), str(date.day))
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
        
        f = open("./Games/%s/%s/%s/%s" % (date.year, date.month, date.day, str(date) + '.txt'), 'w')
        for game in self.allGames:
            f.write(str(game)+'\n')
        f.close()

    def writeError(self, date):
        try:
            path = "./Errors/%s/%s/%s" % (str(date.year), str(date.month), str(date.day))
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
        f = open("./Errors/%s/%s/%s/%s" % (date.year, date.month, date.day, str(date) + '.txt'), 'w')
        f.write("DATE ERROR: " + str(date))
        f.close()



