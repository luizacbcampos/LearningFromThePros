import sqlite3
from PlayerPenaltyEvent import *


class SQL:
    def __init__(self, databaselocation):
        print("Creating new SQL")
        self.databaseLocation = databaselocation
        self.conn = sqlite3.connect(databaselocation)
        self.cur = self.conn.cursor()
        self.conn.text_factory = str
        print("Opened database successfully")


    # Uploads to test database.
    # I am not proud of this code. I am sorry. I was in a rush.
    def addNewPlayerTest(self, ppevent):
        params = (ppevent.getPlayerName(), ppevent.getTeam(), ppevent.getDate(), ppevent.getGameId(), ppevent.getFoot(), ppevent.getOutcome(), ppevent.getDirection(), ppevent.getPenaltyEventString())
        self.cur.execute("INSERT INTO penaltyKickTestTable VALUES (?, ?, ?, ?, ?, ?, ?, ?);" , params)

    def addNewPlayer(self, ppevent):
        params = (ppevent.getPlayerName(), ppevent.getTeam(), ppevent.getDate(), ppevent.getGameId(), ppevent.getFoot(), ppevent.getOutcome(), ppevent.getDirection(), ppevent.getPenaltyEventString())
        self.cur.execute("INSERT INTO penaltyKickTable VALUES (?, ?, ?, ?, ?, ?, ?, ?);" , params)
        # print("Player successfully added")

    def getPlayerData(self, playerName):
        params = playerName
        return self.cur.execute("SELECT * FROM penaltyKickTable WHERE Player = (?) COLLATE NOCASE;", (params,)).fetchall()



        # if playerData > 0:
        #     print "Player Found!"
        #
        #     for penaltyKick in playerData:
        #         print penaltyKick
        # else:
        #     print "Invalid Player!"


    def commitChanges(self, ptr=False):
        self.conn.commit()
        if ptr:
            print("Successfully committed")

    def closeConnection(self):
        self.cur.close()
        self.conn.close()
        print("Connections closed")

    def writeError(self, ppevent):
        f = open("Errors/%s" % ppevent.getPlayerName() + '.txt', 'w')
        f.write("SQL UPLOAD ERROR: " + ppevent.getPlayerName + ppevent.getGameID())
        f.close()


