import os
import regex as re

from bs4 import BeautifulSoup
from PlayerPenaltyEvent import *
from requests_html import HTMLSession


class GameScraper:
    def __init__(self, gameId, date, session):
        self.gameId = gameId
        # self.gameUrl = "http://www.espnfc.us/commentary?gameId=" + str(gameId)
        # self.gameUrl = "https://www.espn.com/soccer/match?gameId="+str(gameId)
        self.gameUrl = "https://www.espn.com/soccer/commentary?gameId="+str(gameId)
        self.session = session
        self.gameDate = date
        self.gameBeautifulSoup = None
        self.allCommentaryPenaltyEvents = None
        self.listOfPlayerPenaltyEvents = []
        self.gameDetails = ''
        # print(self.gameUrl)

    def getGameId(self):
        return self.gameId

    def getGameUrl(self):
        return self.gameUrl

    def getGameBeautifulSoup(self):
        return self.gameBeautifulSoup

    def getAllCommentaryPenaltyEvents(self):
        return self.allCommentaryPenaltyEvents

    def getListOfPlayerPenaltyEvents(self):
        return self.listOfPlayerPenaltyEvents

    def getgameDetails(self):
        if type(self.gameDetails)==str:
            return self.gameDetails
        return ''

    def closeSession(self):
        self.session.close()

    # Make a BeautifulSoup of the game page
    def makeGameBeautifulSoup(self):
        # Must use html5lib to correctly scrape the page. Doesn't work without this.

        headers = {'User-Agent': 
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
        r = self.session.get(self.gameUrl, headers=headers, timeout=1)
        r.html.render()
        self.gameBeautifulSoup = BeautifulSoup(r.html.raw_html, "html.parser")
        # print(self.beautifulSoup.prettify())
        self.makeGameDetails()

    def makeGameDetails(self):
        game_details = self.gameBeautifulSoup.find("div", {"class" :["game-details header"]})
        self.gameDetails = game_details.getText().strip()
        # print(self.gameDetails, ":", self.gameUrl)
        
    def penaltyEventstring(self, penaltyEvent):
        penalty = "Player: {} | Foot: {} | Team: {} | GameID: {} \
            | Outcome: {} | Direction: {}".format(penaltyEvent.getPlayerName(), penaltyEvent.getFoot(), 
                penaltyEvent.getTeam(), str(penaltyEvent.getGameId()).strip(), 
                penaltyEvent.getOutcome(), penaltyEvent.getDirection())
        return penalty

    def printPenaltyEvent(self, penaltyEvent):
        print(self.penaltyEventstring(penaltyEvent))
           

    def printListOfPlayerPenaltyEvents(self):
        print("Total number of penalties: {}".format(len(self.listOfPlayerPenaltyEvents)))
        
        for penaltyEvent in self.listOfPlayerPenaltyEvents:
            self.printPenaltyEvent(penaltyEvent)


    def makeAllCommentaryPenaltyEvents(self):
        wrap = self.gameBeautifulSoup.find("div", id="gamepackage-wrap")
        tabs = wrap.find("div", id="gamepackage-soccer-commentary-tabs")
        comment = tabs.find("div", id="match-commentary-1-tab-1")
        table = comment.find("table")
        events = table.find_all("td", class_="game-details")
        
        # print(events)
        # print("\nPenalties:\n")
        self.allCommentaryPenaltyEvents = table.find_all(text = re.compile("(?i)penalty"), class_="game-details")
        # print(self.allCommentaryPenaltyEvents)


    def makeListOfPlayerPenaltyEvents(self):
        
        allPenaltyEvents = []
        for penaltyEvent in self.allCommentaryPenaltyEvents:
            allPenaltyEvents.append(str(penaltyEvent))

        # /TODO figure out why this duplicates
        # allPenaltyEvents = allPenaltyEvents[:-1]

        # get the player name
        for penaltyEvent in allPenaltyEvents:
            # added because going back to 2015 they seemed to use 'penalty area' rather than box.
            if "penalty area" in penaltyEvent:
                continue

            elif "Penalty saved!" in penaltyEvent:
                # This is a Penalty Saved
                saveFeatures = penaltyEvent.split("!")[-1][:-17]
                playerName = saveFeatures.split("(")[0][1:-1]
                currentPlayer = PlayerPenaltyEvent(playerName)
                currentPlayer.setFoot(re.findall("\w+(?=\s*foot[^/])", saveFeatures)[0])
                currentPlayer.setTeam(saveFeatures[saveFeatures.find("(") + 1:saveFeatures.find(")")])
                currentPlayer.setGameId(self.gameId)
                currentPlayer.setOutcome(0)
                direction = saveFeatures.split("in the")[-1]
                currentPlayer.setDirection(direction)
                currentPlayer.setPenaltyEventString(penaltyEvent)
                currentPlayer.setDate(self.gameDate)
                self.listOfPlayerPenaltyEvents.append(currentPlayer)

            elif "Penalty missed!" in penaltyEvent:
                # This is penalty missed
                if ("Bad penalty by" in penaltyEvent):
                    missedFeatures = penaltyEvent.split("Bad penalty by")[-1]
                    playerName = missedFeatures.split("(")[0][1:-1]
                    currentPlayer = PlayerPenaltyEvent(playerName)
                    currentPlayer.setFoot(re.findall("\w+(?=\s*foot[^/])", missedFeatures)[0])
                    currentPlayer.setTeam(missedFeatures[missedFeatures.find("(") + 1:missedFeatures.find(")")])
                    currentPlayer.setGameId(self.gameId)
                    currentPlayer.setOutcome(0)
                    if ("post" in missedFeatures):
                        currentPlayer.setDirection(re.findall("\w+(?=\s*post[^/])", missedFeatures)[0] + " post")

                    elif ("is close" in missedFeatures):
                        direction = missedFeatures.split("to the")[-1].split(".")[0]
                        currentPlayer.setDirection(direction)

                    elif ("misses" in missedFeatures):
                        direction = missedFeatures.split("to the")[-1].split(".")[0]
                        currentPlayer.setDirection(direction)
                    else:
                        direction = re.findall("^.* is (.*)\. ", missedFeatures)[0]
                        currentPlayer.setDirection(direction)
                    currentPlayer.setPenaltyEventString(penaltyEvent)
                    currentPlayer.setDate(self.gameDate)
                    self.listOfPlayerPenaltyEvents.append(currentPlayer)
                # Weird thing where if it hits a post for some reason it seems to do a different type of message. Idk why
                ## /TODO: fix the post error handling if it is true that this only happens during posts.
                else:
                    match = re.search("[0-9]\. ((?:(?!\.).)*)", penaltyEvent)
                    if match:
                        missedFeatures = penaltyEvent[2 + match.start():] + penaltyEvent[:match.end()]
                    else:
                        missedFeatures = penaltyEvent.split(").")[-1]
                    playerName = missedFeatures.split("(")[0][1:-1]
                    currentPlayer = PlayerPenaltyEvent(playerName)
                    currentPlayer.setFoot(re.findall("\w+(?=\s*foot[^/])", missedFeatures)[0])
                    currentPlayer.setTeam(missedFeatures[missedFeatures.find("(") + 1:missedFeatures.find(")")])
                    currentPlayer.setGameId(self.gameId)
                    currentPlayer.setOutcome(0)
                    if ("post" in missedFeatures):
                         currentPlayer.setDirection(re.findall("\w+(?=\s*post[^/])", missedFeatures)[0] + " post")
                    elif ("bar" in missedFeatures):
                        currentPlayer.setDirection(re.findall("\w+(?=\s*bar[^/])", missedFeatures)[0] + " bar")
                    elif ("is close" in missedFeatures):
                        direction = missedFeatures.split("to the")[-1].split(".")[0]
                        currentPlayer.setDirection(direction)
                    else:
                        direction = re.findall("^.* is (.*)\. ", missedFeatures)[0]
                        currentPlayer.setDirection(direction)
                    currentPlayer.setPenaltyEventString(penaltyEvent)
                    currentPlayer.setDate(self.gameDate)
                    self.listOfPlayerPenaltyEvents.append(currentPlayer)

            elif "Goal!" in penaltyEvent and "penalty" in penaltyEvent:
                goalFeatures = penaltyEvent.split(".")[1]
                try:
                    playerName = goalFeatures.split("(")[0][1:-1]
                    currentPlayer = PlayerPenaltyEvent(playerName)
                    currentPlayer.setFoot(re.findall("\w+(?=\s*foot[^/])", goalFeatures)[0])
                    currentPlayer.setTeam(goalFeatures[goalFeatures.find("(") + 1:goalFeatures.find(")")])
                    currentPlayer.setGameId(self.gameId)
                    currentPlayer.setOutcome(1)
                    direction = goalFeatures.split("to the")[-1]
                    currentPlayer.setDirection(direction)
                    currentPlayer.setPenaltyEventString(penaltyEvent)
                    currentPlayer.setDate(self.gameDate)
                    self.listOfPlayerPenaltyEvents.append(currentPlayer)
                except:
                    if re.search("(\w [0-9]\. (.*)\.)", penaltyEvent, overlapped=True):
                        goalFeatures = re.search("(\w [0-9]\. (.*)\.)", penaltyEvent, overlapped=True).group()[4:]
                    elif re.search("[0-9]\)\. (.*)\.", penaltyEvent, overlapped=True):
                        goalFeatures = re.search("([0-9]\)\. (.*)\.)", penaltyEvent, overlapped=True).group()[3:]
                    else:
                        goalFeatures = re.search("([A-z]\. [0-9]\. (.*)\.)", penaltyEvent, overlapped=True).group()[5:]
                    playerName = goalFeatures.split("(")[0][1:-1]
                    currentPlayer = PlayerPenaltyEvent(playerName)
                    currentPlayer.setFoot(re.findall("\w+(?=\s*foot[^/])", goalFeatures)[0])
                    currentPlayer.setTeam(goalFeatures[goalFeatures.find("(") + 1:goalFeatures.find(")")])
                    currentPlayer.setGameId(self.gameId)
                    currentPlayer.setOutcome(1)
                    direction = goalFeatures.split("to the")[-1]
                    currentPlayer.setDirection(direction)
                    currentPlayer.setPenaltyEventString(penaltyEvent)
                    currentPlayer.setDate(self.gameDate)
                    self.listOfPlayerPenaltyEvents.append(currentPlayer)

            elif "penalty kick" in penaltyEvent:
                playerName = re.search("(\w+\s\w+)[\w\s]+penalty kick", penaltyEvent).group(1)
                currentPlayer = PlayerPenaltyEvent(playerName)
                currentPlayer.setFoot(re.findall("\w+(?=\s*foot)", penaltyEvent)[0])
                currentPlayer.setTeam(None)
                currentPlayer.setGameId(self.gameId)
                if "scores" in penaltyEvent:
                    currentPlayer.setOutcome(1)
                else:
                    currentPlayer.setOutcome(0)
                if "crossbar" in penaltyEvent:
                    if "hits" in penaltyEvent:
                        currentPlayer.setDirection("the bar")
                    else:
                        currentPlayer.setDirection("too high")

                elif "wide" in penaltyEvent:
                    currentPlayer.setDirection(re.search("(?<=wide\s)(\w+)", penaltyEvent).group(1))
                else:
                    currentPlayer.setDirection(re.search("(?<=foot)(.*)(?= and )", penaltyEvent).group(1))
                currentPlayer.setPenaltyEventString(penaltyEvent)
                currentPlayer.setDate(self.gameDate)
                self.listOfPlayerPenaltyEvents.append(currentPlayer)




    def writeError(self, date):
        #Write the game Page HTML to a file
        try:
            path = "./GameErrors/%s/%s/%s" % (str(date.year), str(date.month), str(date.day))
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
        f = open("./GameErrors/%s/%s/%s/%s" % (str(date.year), str(date.month), str(date.day), self.gameId + '.txt'), 'w')
        f.write("GAME ERROR: " + self.gameId)
        f.close()

    def writePenalty(self, date):
        f = open("./Penaties_%s" % (str(date.year)+'.txt'), 'a+')

        for penaltyEvent in self.listOfPlayerPenaltyEvents:
            f.write("Date: "+ str(date) +" | ")
            f.write(self.penaltyEventstring(penaltyEvent))
            f.write('\n')

        f.close()