from DateScraper import *
from GameScraper import *
from SQL import *
from datetime import *
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# from Tkinter import *
# import Tkinter as Tk
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def main():
    print("Hello World.")
    #dataDownloader(datetime(2017,12, 10), datetime(2017,12, 12))
    drawPenaltyHistory("Lionel Messi")


def sumPenaltyLocations(penalties):
    penaltyMadeDict = {'left': 0, 'high': 0,'right': 0,'leftPost': 0, 'rightPost': 0, 'bar': 0, 'bottomLeftCorner': 0,
                   'topLeftCorner': 0, 'bottomRightCorner': 0, 'topRightCorner': 0, 'highCenter': 0, 'center': 0
                   }
    penaltyMissedDict = {'left': 0, 'high': 0, 'right': 0, 'leftPost': 0, 'rightPost': 0, 'bar': 0, 'bottomLeftCorner': 0, 
                    'topLeftCorner': 0, 'bottomRightCorner': 0, 'topRightCorner': 0, 'highCenter': 0, 'center': 0
                    }

    for penalty in penalties:
        penaltyOutcome = penalty.__getitem__(5)
        penaltyLocation = penalty.__getitem__(6).replace(" ", "").replace(".", "").replace("tothe", "").replace("lower", "bottom")
        if penaltyOutcome == "1":
            if penaltyLocation == "bottomleftcorner":
                penaltyMadeDict['bottomLeftCorner'] += 1
            elif penaltyLocation == "topleftcorner":
                penaltyMadeDict['topLeftCorner'] += 1
            elif penaltyLocation == "bottomrightcorner":
                penaltyMadeDict["bottomRightCorner"] += 1
            elif penaltyLocation == "toprightcorner":
                penaltyMadeDict["topRightCorner"] += 1
            elif "highcentre" in penaltyLocation:
                penaltyMadeDict['highCenter'] += 1
            else:
                print("MADE:     " + penaltyLocation)
                penaltyMadeDict["center"] += 1
        else:
            if "toohigh" in penaltyLocation:
                penaltyMissedDict['high'] += 1
            elif penaltyLocation == "right":
                penaltyMissedDict['right'] += 1
            elif penaltyLocation == "left":
                penaltyMissedDict['left'] += 1
            elif penaltyLocation == "thebar":
                penaltyMissedDict['bar'] += 1
            elif penaltyLocation == "leftpost":
                penaltyMissedDict['leftPost'] += 1
            elif penaltyLocation == "rightpost":
                penaltyMissedDict['rightPost'] += 1
            elif penaltyLocation == "bottomleftcorner":
                penaltyMissedDict['bottomLeftCorner'] += 1
            elif penaltyLocation == "topleftcorner":
                penaltyMissedDict['topLeftCorner'] += 1
            elif penaltyLocation == "bottomrightcorner":
                penaltyMissedDict["bottomRightCorner"] += 1
            elif penaltyLocation == "toprightcorner":
                penaltyMissedDict["topRightCorner"] += 1
            elif "highcentre" in penaltyLocation:
                penaltyMissedDict['highCenter'] += 1
            else:
                print("MISSED:   " + penaltyLocation)
                penaltyMissedDict["center"] += 1

    return [penaltyMadeDict, penaltyMissedDict]

def drawPenaltyHistory(playerName):
    sqlDownload = SQL("penaltyKicks.db")
    playerPenalties = sqlDownload.getPlayerData(playerName)
    penaltyData = sumPenaltyLocations(playerPenalties)
    print("----------------------------------")
    madePenalties = penaltyData[0]
    missedPenalties = penaltyData[1]
    print(penaltyData)

    hPositions = 5
    vPositions = 7


    # matrix = [["", "", "", madePenalties['high'], "", "", ""],
    #           ["", "", "", madePenalties['bar'], "", "", ""],
    #           ["", "", madePenalties['topLeftCorner'], madePenalties['highCenter'], madePenalties["topRightCorner"], "",
    #            ""],
    #           [madePenalties['left'], madePenalties['leftPost'], "", madePenalties['center'], "",
    #            madePenalties['rightPost'], madePenalties['right']],
    #           ["", "", madePenalties['bottomLeftCorner'], "", madePenalties['bottomRightCorner'], "", ""]]

    matrix = [["", "", "", missedPenalties['high'], "", "", ""],
              ["", "", "", missedPenalties['bar'], "", "", ""],
              ["", "", str(madePenalties['topLeftCorner']) + "\nSaved: " + str(missedPenalties['topLeftCorner']),
               str(madePenalties['highCenter']) + "\nSaved: " + str(missedPenalties['highCenter']),
               str(madePenalties["topRightCorner"]) + "\nSaved: " + str(missedPenalties['topRightCorner']), "",
               ""],
              [missedPenalties['left'], missedPenalties['leftPost'], "", str(madePenalties['center']) + "\nSaved: " + str(missedPenalties["center"]), "",
               missedPenalties['rightPost'], missedPenalties['right']],
              ["", "", str(madePenalties['bottomLeftCorner']) + "\nSaved: " + str(missedPenalties['bottomLeftCorner']), "", str(madePenalties['bottomRightCorner']) + "\nSaved: " + str(missedPenalties['bottomRightCorner']), "", ""]]

    f = plt.figure()
    plt.title(playerName)
    tb = plt.table(cellText=matrix, loc=(0, 0), cellLoc='center')

    tc = tb.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0 / hPositions)
        cell.set_width(1.0 / vPositions)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.show()

    root = Tk.Tk()
    root.wm_title("Penalty Visualization")

    Label(root, text="Start Date").grid(row=0)
    Label(root, text="End Date").grid(row=1)
    startDate = Entry(root)
    endDate = Entry(root)

    startDate.grid(row=0, column=1)
    endDate.grid(row=1, column=1)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.LEFT)
    Tk.mainloop()


# Date aux
def dateGenerator(start, end):
    '''
        Generates dates from [start,finish]
    '''
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

def date_from_list(lista):
    '''
        Generates dates from list
    '''
    for dt in lista:
        dt = dt.split("-")
        dt = date(int(dt[0]), int(dt[1]), int(dt[2]))
        yield dt

def date_from_line(line):
    '''
        Parses line to get a date
    '''
    dt = line.split("/")[-1].split(".")[0].split("-")
    dt = date(int(dt[0]), int(dt[1]), int(dt[2]))
    return dt


# Downloders

def getCurrentDay(dt):
    '''
        Scraps ESPN for games in a date
    '''
    
    currentDate = str(dt).replace("-", "")
    print("Beginning ESPN Scrape of day: " + currentDate, end=' -> ')
    currentDay = DateScraper(currentDate)

    try:
        currentDay.makeBeautifulSoup()
        currentDay.makeListOfgames()
        currentDay.writeGameList(dt)

    except Exception as e:
        # raise e
        print("Error in DATE Scraping: " + currentDay.getDate())
        currentDay.writeError(dt)
    
    return currentDay

def fill_currentGame(sqlUpload, dt, currentGame, use_tqdm=False, bar=None):
    '''
        Gets game information
    '''
    try:
        currentGame.makeGameBeautifulSoup()
    except Exception as e:
        if use_tqdm:
            bar.write("Error in soup GAME Scraping: %s" % currentGame.gameId)
        else:
            print("Error in soup GAME Scraping: " + currentGame.gameId)
        currentGame.writeError(dt)

    if "English Premier League" in currentGame.getgameDetails(): #continue
        # print("VALID GAME:"+str(currentGame.getGameId()))
        try:
            currentGame.makeAllCommentaryPenaltyEvents()
            currentGame.makeListOfPlayerPenaltyEvents()
        except Exception as e:
            if use_tqdm:
                bar.write("Error in GAME Scraping: %s" % currentGame.gameId)
            else:
                print("Error in GAME Scraping: " + currentGame.gameId)
            currentGame.writeError(dt)

        currentGamePenalties = currentGame.getListOfPlayerPenaltyEvents()
        
        for eachPlayer in currentGamePenalties:
            try:
                currentGame.writePenalty(dt)
                sqlUpload.addNewPlayer(eachPlayer)
            except:
                print("SQL UPLOAD ERROR")

def dataDownloader(date1, date2):
    '''
        Original data downloader. Downloads all at once
    '''
    sqlUpload = SQL("penaltyKicks.db")

    for dt in dateGenerator(date1, date2):
        currentDate = str(dt).replace("-", "")
        currentDay = getCurrentDay(dt)

        all_games = currentDay.getAllGames()
        session = currentDay.getSession()

        print("amount of games: ", currentDay.getAmountGames())

        for gameID in all_games:
            print(gameID)
            currentGame = GameScraper(gameID, currentDate, session)
            fill_currentGame(sqlUpload, dt, currentGame, use_tqdm=False, bar=None)

        currentDay.closeSession()
        sqlUpload.commitChanges()

    sqlUpload.closeConnection()

def download_game_from_day_list(day_list):

    sqlUpload = SQL("penaltyKicks.db")    
    for dt in date_from_list(day_list):
        currentDate = str(dt).replace("-", "")
        pbar.set_description("Day %s" % currentDate)
        
        # currentDay = DateScraper(currentDate)

    sqlUpload.closeConnection()
    return

# Files reading
def read_erros(file="erros.txt"):
    '''
        Try the dates in file again
    '''
    f = open(file, "r")
    for line in f.readlines():
        data = line.strip().split("-")
        currentDay = getCurrentDay(date(int(data[0]), int(data[1]), int(data[2])))
        currentDay.closeSession()

def read_game_errors():
    '''
        Try again games that failed
    '''
    import glob

    sqlUpload = SQL("penaltyKicks.db")
    pbar = tqdm(glob.glob('GameErrors/*/*/*/*.txt'))
    
    for name in pbar:
        Lname = name.split('/')
        dt = date(int(Lname[1]), int(Lname[2]), int(Lname[3]))

        currentDate = str(dt).replace("-", "")
        pbar.set_description("Day %s" % currentDate)
        
        currentDay = DateScraper(currentDate)
        session = currentDay.getSession()

        gameID = Lname[-1].split('.')[0]

        currentGame = GameScraper(gameID, currentDate, session)
        fill_currentGame(sqlUpload, dt, currentGame, use_tqdm=True, bar=pbar)
    
        currentDay.closeSession()
        sqlUpload.commitChanges()

    sqlUpload.closeConnection()

def read_games(year='2020'):
    '''
        Get data from games from year
    '''
    import glob

    sqlUpload = SQL("penaltyKicks.db")

    pbar = tqdm(glob.glob('Games/{}/*/*/*.txt'.format(year)))
    for name in pbar:
        dt = date_from_line(name)
        currentDate = str(dt).replace("-", "")
        pbar.set_description("Day %s" % currentDate)
        
        currentDay = DateScraper(currentDate)
        session = currentDay.getSession()
        # print("Day: " + currentDate)
        
        f = open(name, "r")

        progressbar = tqdm(f.readlines(), leave=False)
        for line in progressbar:
            gameID = line.strip()
            progressbar.set_description("Game %s" % gameID)
            # print("\tGame: "+ str(gameID), end=' -> ')
            currentGame = GameScraper(gameID, currentDate, session)
            fill_currentGame(sqlUpload, dt, currentGame, use_tqdm=True, bar=progressbar)
        f.close()
        # break

        currentDay.closeSession()
        sqlUpload.commitChanges()

    sqlUpload.closeConnection()


# Tests
def TestGameScraper(gameID):
    sqlUpload = SQL("penaltyKicks.db")
    for dt in dateGenerator(date(2018, 1, 1), date(2018, 1, 1)):
        currentDate = str(dt).replace("-", "")
        currentDay = DateScraper(currentDate)

        session = currentDay.getSession()
        currentGame = GameScraper(gameID, currentDate, session)
        fill_currentGame(sqlUpload, dt, currentGame)

        currentGame.printListOfPlayerPenaltyEvents()

if __name__ == "__main__":

    # dataDownloader(date(2018, 1, 1), date(2021, 7, 31))
    # dataDownloader(date(2018, 9, 10), date(2021, 7, 31))
    
    # TestGameScraper('605706')
    # TestGameScraper('480691')

    # read_erros()
    read_games('2019')

    # read_game_errors()
    # main()

    day_list = ['2020-03-07', '2020-09-13', '2020-09-14', '2020-09-21', '2020-09-27', '2020-10-18', '2020-11-01', 
    '2020-12-19', '2020-12-21']
    download_game_from_day_list(day_list)