from DateScraper import *
from GameScraper import *
from SQL import *
from datetime import *

import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# from Tkinter import *
# import Tkinter as Tk
import glob
from tqdm import tqdm
from os.path import exists

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


# --- Date aux ---

def dateGenerator(**kwargs):
    
    if 'lista' in kwargs:
        '''
        Generates dates from list
        '''
        for dt in kwargs['lista']:
            dt = dt.split("-")
            dt = date(int(dt[0]), int(dt[1]), int(dt[2]))
            yield dt
    else:
        '''
        Generates dates from [start,finish]
        '''
        current = kwargs['start']
        while current <= kwargs['end']:
            yield current
            current += timedelta(days=1)
                
def date_from_line(line):
    '''
        Parses line to get a date
    '''
    dt = line.split("/")[-1].split(".")[0].split("-")
    dt = date(int(dt[0]), int(dt[1]), int(dt[2]))
    return dt

def fetched_date(dt):
    '''
        Returns whether or not a Date was scrapped
    '''

    file_path = 'Games/{}/{}/{}/{}.txt'.format(str(dt.year), str(dt.month), str(dt.day), str(dt))
    return exists(file_path)

def games_from_date(dt):
    '''
        Returns all games from a fetched date
    '''
    file_path = 'Games/{}/{}/{}/{}.txt'.format(str(dt.year), str(dt.month), str(dt.day), str(dt))
    with open(file_path, 'r') as f:
        all_games = [line.strip() for line in f.readlines()]
    return all_games

# Games

def files_from_year(year='2020'):
    return glob.glob('Games/{}/*/*/*.txt'.format(year))

def failed_files():
    return glob.glob('GameErrors/*/*/*/*.txt')

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

    for dt in dateGenerator(start=date1, end=date2):
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

def Download_data(**kwargs):

    sqlUpload = SQL("penaltyKicks.db")

    for dt in dateGenerator(**kwargs):
        print(dt)
        currentDate = str(dt).replace("-", "")

        if fetched_date(dt):
            currentDay = DateScraper(currentDate)
            all_games = games_from_date(dt)
        else:
            currentDay = getCurrentDay(dt)
            all_games = currentDay.getAllGames()

        session = currentDay.getSession()

        if all_games is not None:
            progressbar = tqdm(all_games, leave=False)
            for gameID in progressbar:
                progressbar.set_description("Game %s" % gameID)
                currentGame = GameScraper(gameID, currentDate, session)
                fill_currentGame(sqlUpload, dt, currentGame, use_tqdm=True, bar=progressbar)

        currentDay.closeSession()
        sqlUpload.commitChanges()
    
    sqlUpload.closeConnection()

# --- Files reading --- 
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

    sqlUpload = SQL("penaltyKicks.db")
    pbar = tqdm(failed_files())
    
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
    sqlUpload = SQL("penaltyKicks.db")

    pbar = tqdm(files_from_year(year))
    for name in pbar:
        dt = date_from_line(name)
        currentDate = str(dt).replace("-", "")
        pbar.set_description("Day %s" % currentDate)
        
        currentDay = DateScraper(currentDate)
        session = currentDay.getSession()
        # print("Day: " + currentDate)
        progressbar = tqdm(games_from_date(dt), leave=False)
        for line in progressbar:
            gameID = line.strip()
            progressbar.set_description("Game %s" % gameID)
            # print("\tGame: "+ str(gameID), end=' -> ')
            currentGame = GameScraper(gameID, currentDate, session)
            fill_currentGame(sqlUpload, dt, currentGame, use_tqdm=True, bar=progressbar)

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
    # read_games('2019')

    # read_game_errors()
    # main()

    season17_18 = ['2017-08-11', '2017-08-12', '2017-08-13', '2017-08-19', '2017-08-20', '2017-08-21', '2017-08-26', '2017-08-27', '2017-09-09', '2017-09-10', '2017-09-11', '2017-09-15', '2017-09-16', '2017-09-17', '2017-09-23', '2017-09-24', '2017-09-25', '2017-09-30', '2017-10-01', '2017-10-14', '2017-10-15', '2017-10-16', '2017-10-20', '2017-10-21', '2017-10-22', '2017-10-23', '2017-10-28', '2017-10-29', '2017-10-30', '2017-11-04', '2017-11-05', '2017-11-18', '2017-11-19', '2017-11-20', '2017-11-24', '2017-11-25', '2017-11-26', '2017-11-28', '2017-11-29', '2017-12-02', '2017-12-03', '2017-12-09', '2017-12-10', '2017-12-12', '2017-12-13', '2017-12-16', '2017-12-17', '2017-12-18', '2017-12-22', '2017-12-23', '2017-12-26', '2017-12-27', '2017-12-28', '2017-12-30', '2017-12-31', '2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-13', '2018-01-14', '2018-01-15', '2018-01-20', '2018-01-21', '2018-01-22', '2018-01-30', '2018-01-31', '2018-02-03', '2018-02-04', '2018-02-05', '2018-02-10', '2018-02-11', '2018-02-12', '2018-02-24', '2018-02-25', '2018-03-01', '2018-03-03', '2018-03-04', '2018-03-05', '2018-03-10', '2018-03-11', '2018-03-12', '2018-03-16', '2018-03-17', '2018-03-31', '2018-04-01', '2018-04-07', '2018-04-14', '2018-04-21', '2018-04-28', '2018-05-13', '2018-05-05', ]
    season18_19 = ['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-12', '2019-01-13', '2019-01-14', '2019-01-19', '2019-01-20', '2019-01-29', '2019-01-30', '2019-02-02', '2019-02-03', '2019-02-04', '2019-02-06', '2019-02-09', '2019-02-10', '2019-02-11', '2019-02-22', '2019-02-23', '2019-02-24', '2019-02-26', '2019-02-27', '2019-03-02', '2019-03-03', '2019-03-09', '2019-03-10', '2019-03-16', '2019-03-17', '2019-03-30', '2019-03-31', '2019-04-01', '2019-04-05', '2019-04-06', '2019-04-07', '2019-04-08', '2019-04-12', '2019-04-13', '2019-04-14', '2019-04-15', '2019-04-16', '2019-04-20', '2019-04-21', '2019-04-22', '2019-04-23', '2019-04-24', '2019-04-26', '2019-04-27', '2019-04-28', '2019-05-03', '2019-05-04', '2019-05-05', '2019-05-06', '2019-05-12', '2018-08-10', '2018-08-11', '2018-08-12', '2018-08-18', '2018-08-19', '2018-08-20', '2018-08-25', '2018-08-26', '2018-08-27', '2018-09-01', '2018-09-02', '2018-09-15', '2018-09-16', '2018-09-17', '2018-09-22', '2018-09-23', '2018-09-29', '2018-09-30', '2018-10-01', '2018-10-05', '2018-10-06', '2018-10-07', '2018-10-20', '2018-10-21', '2018-10-27', '2018-10-28', '2018-10-29', '2018-11-03', '2018-11-04', '2018-11-05', '2018-11-10', '2018-11-11', '2018-11-24', '2018-11-25', '2018-11-26', '2018-11-30', '2018-12-01', '2018-12-02', '2018-12-04', '2018-12-05', '2018-12-08', '2018-12-09', '2018-12-10', '2018-12-15', '2018-12-16', '2018-12-21', '2018-12-22', '2018-12-23', '2018-12-26', '2018-12-27', '2018-12-29', '2018-12-30', ]
    season19_20 = ['2019-08-10', '2019-08-17', '2019-08-24', '2019-09-14', '2019-09-21', '2019-09-28', '2019-10-05', '2019-10-19', '2019-11-02', '2019-12-22']
    
    day_list = []

    Download_data(lista=season19_20)

    # Download_data(lista=season18_19)