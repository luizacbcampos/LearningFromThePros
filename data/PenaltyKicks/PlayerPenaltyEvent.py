class PlayerPenaltyEvent:
    def __init__(self, playername):
        self.playerName = playername
        self.team = None
        self.date = None
        self.gameId = None
        self.foot = None
        self.outcome = None
        self.direction = None
        self.penaltyEventString = None


    def getPlayerName(self):
        return self.playerName

    def getTeam(self):
        return self.team

    def getDate(self):
        return self.date

    def getGameId(self):
        return self.gameId

    def getFoot(self):
        return self.foot

    def getOutcome(self):
        return self.outcome

    def getDirection(self):
        return self.direction

    def getPenaltyEventString(self):
        return self.penaltyEventString

    def setPlayeName(self, playerName):
        self.playerName = playerName

    def setTeam(self, country):
        self.team = country

    def setDate(self, date):
        self.date = date

    def setGameId(self, gameId):
        self.gameId = gameId

    def setFoot(self, foot):
        self.foot = foot

    def setOutcome(self, penalty):
        self.outcome = penalty

    def setDirection(self, direction):
        self.direction = direction

    def setPenaltyEventString(self, penaltyEventString):
        self.penaltyEventString = penaltyEventString



