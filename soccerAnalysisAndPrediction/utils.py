import json
from tqdm import tqdm
from collections import Counter
import numpy as np
import operator
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
import networkx as nx
import base64
from collections import defaultdict
import sys,os
import math
import random
import operator
import csv
import matplotlib.pylab as pyl
import itertools
import scipy as sp
from scipy import stats
from scipy import optimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotsoccer
import socceraction.spadl as spadl
import socceraction.spadl.statsbomb as statsbomb
from io import StringIO
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from pandas.io.json import json_normalize
from .FCPython import createPitch


ACCURATE_PASS = 1801
EVENT_TYPES = ['Duel', 'Foul', 
             'Offside', 'Shot']

TOURNAMENTS=['Italy','England','Germany', 'France', 
             'Spain', 'European_Championship','World_Cup']

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_folder = os.path.join(BASE_DIR, 'static/data/')

# data_folder='data/'
def load_public_dataset(data_folder=data_folder, tournament='Italy'):
    """
    Load the json files with the matches, events, players and competitions
    
    Parameters
    ----------
    data_folder : str, optional
        the path to the folder where json files are stored. Default: 'data/'
        
    tournaments : list, optional
        the list of tournaments to load. 
        
    Returns
    -------
    tuple
        a tuple of four dictionaries, containing matches, events, players and competitions
        
    """
    # loading the matches and events data
    matches, events = {}, {}
    with open(data_folder + '/events/events_%s.json' %tournament) as json_data:
    # with open('./data/events/events_%s.json' %tournament) as json_data:
        events = json.load(json_data)
    # with open('./data/matches/matches_%s.json' %tournament) as json_data:
    with open(data_folder + '/matches/matches_%s.json' %tournament) as json_data:
        matches = json.load(json_data)
    
    match_id2events = defaultdict(list)
    match_id2match = defaultdict(dict)
    for event in events:
        match_id = event['matchId']
        match_id2events[match_id].append(event)
                                         
    for match in matches:
        match_id = match['wyId']
        match_id2match[match_id] = match
                                   
    # loading the players data
    # with open('./data/players.json') as json_data:
    with open(data_folder + '/players.json') as json_data:
        players = json.load(json_data)
    
    player_id2player = defaultdict(dict)
    for player in players:
        player_id = player['wyId']
        player_id2player[player_id] = player
    
    # loading the competitions data
    competitions={}
    # with open('./data/competitions.json') as json_data:
    with open(data_folder + '/competitions.json') as json_data:
        competitions = json.load(json_data)
    competition_id2competition = defaultdict(dict)
    for competition in competitions:
        competition_id = competition['wyId']
        competition_id2competition[competition_id] = competition
    
    # loading the competitions data
    teams={}
    # with open('./data/teams.json') as json_data:
    with open(data_folder + '/teams.json') as json_data:
        teams = json.load(json_data)
    team_id2team = defaultdict(dict)
    for team in teams:
        team_id = team['wyId']
        team_id2team[team_id] = team
    
    return match_id2match, match_id2events, player_id2player, competition_id2competition, team_id2team

def load_Statsbomb_dataset(data_folder=data_folder + "Statsbomb"):
    # loading the matches and events data
    competitions, matchs, match_events = {}, {}, {}
    with open(data_folder + '/data/competitions.json') as json_data:
    # with open('./data/events/events_%s.json' %tournament) as json_data:
        competitions = json.load(json_data)
    
    competition_ids = os.listdir(data_folder + '/data/matches')
    
    for competition_id in competition_ids:
        matchs[competition_id] = {}
        season_ids =  os.listdir(data_folder + '/data/matches/' + competition_id)
        for season_id in season_ids:
            # with open(data_folder + '\\data\\matches\\' + competition_id + '\\' + season_id ) as json_data:
            with open(data_folder + '/data/matches/' + competition_id + '/' + season_id, encoding="utf-8") as json_data:
                # matchs[competition_id][season_id] = json.load(json_data)
                season_id = str(season_id).replace('.json', '')
                matchs[competition_id][season_id] = json.load(json_data)

    # match_event_files =  os.listdir(data_folder + '/data/events')

    # for match_event_file in match_event_files:
    #     with open(data_folder + '/data/events/' + match_event_file, encoding="utf-8") as json_data:
    #         match_id = str(match_event_file).replace('.json', '')
    #         match_events[match_id] = json.load(json_data)


    # print(matchs)


    # with open(data_folder + '/data/events/events_%s.json' %tournament) as json_data:
    # # with open('./data/events/events_%s.json' %tournament) as json_data:
    #     events = json.load(json_data)

    return competitions, matchs, match_events

def get_weight(position):
    """
    Get the probability of scoring a goal given the position of the field where 
    the event is generated.
    
    Parameters
    ----------
    position: tuple
        the x,y coordinates of the event
    """
    x, y = position
    
    # 0.01
    if x >= 65 and x <= 75:
        return 0.01
    
    # 0.5
    if (x > 75 and x <= 85) and (y >= 15 and y <= 85):
        return 0.5
    if x > 85 and (y >= 15 and y <= 25) or (y >= 75 and y <= 85):
        return 0.5
    
    # 0.02
    if x > 75 and (y <= 15 or y >= 85):
        return 0.02
    
    # 1.0
    if x > 85 and (y >= 40 and y <= 60):
        return 1.0
    
    # 0.8
    if x > 85 and (y >= 25 and y <= 40 or y >= 60 and y <= 85):
        return 0.8
    
    return 0.0

def in_window(events_match, time_window):
    start, end = events_match[0], events[-1]
    return start['eventSec'] >= time_window[0] and end['eventSec'] <= time_window[1]

def segno(x):
    """
    Input:  x, a number
    Return:  1.0  if x>0,
            -1.0  if x<0,
             0.0  if x==0
    """
    if   x  > 0.0: return 1.0
    elif x  < 0.0: return -1.0
    elif x == 0.0: return 0.0

def standard_dev(list):
    ll = len(list)
    m = 1.0 * sum(list)/ll
    return ( sum([(elem-m)**2.0 for elem in list]) / ll )**0.5

def list_check(lista):
    """
    If a list has only one element, return that element. Otherwise return the whole list.
    """
    try:
        e2 = lista[1]
        return lista
    except IndexError:
        return lista[0]
    
def get_event_name(event):
    event_name = ''
    try:
        if event['subEventName'] != '':
            event_name = event_names_df[(event_names_df.event == event['eventName']) & (event_names_df.subevent == event['subEventName'])].subevent_label.values[0]
        else:
            event_name = event_names_df[event_names_df.event == event['eventName']].event_label.values[0]
    except TypeError:
        #print event
        pass
    
    return event_name
    
def is_in_match(player_id, match):
    team_ids = list(match['teamsData'].keys())
    all_players = []
    for team in team_ids:
        in_bench_players = [m['playerId'] for m in match['teamsData'][team]['formation']['bench']]
        in_lineup_players = [m['playerId'] for m in match['teamsData'][team]['formation']['lineup']]
        substituting_players = [m['playerIn'] for m in match['teamsData'][team]['formation']['substitutions']]
        all_players += in_bench_players + in_lineup_players + substituting_players
    return player_id in all_players

def data_download():
    """
    Downloading script for soccer logs public open dataset:
    https://figshare.com/collections/Soccer_match_event_dataset/4415000/2
    Data description available here:
    Please cite the source as:
    Pappalardo, L., Cintia, P., Rossi, A. et al. A public data set of spatio-temporal match events in soccer competitions. 
    Scientific Data 6, 236 (2019) doi:10.1038/s41597-019-0247-7, https://www.nature.com/articles/s41597-019-0247-7
    """

    import requests, zipfile, json, io


    dataset_links = {

    'matches' : 'https://ndownloader.figshare.com/files/14464622',
    'events' : 'https://ndownloader.figshare.com/files/14464685',
    'players' : 'https://ndownloader.figshare.com/files/15073721',
    'teams': 'https://ndownloader.figshare.com/files/15073697',
    'competitions': 'https://ndownloader.figshare.com/files/15073685'
    }

    print ("Downloading matches data")
    r = requests.get(dataset_links['matches'], stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("data/matches")

    
    print ("Downloading teams data")
    r = requests.get(dataset_links['teams'], stream=False)
    print (r.text, file=open('data/teams.json','w'))


    print ("Downloading players data")
    r = requests.get(dataset_links['players'], stream=False)
    print (r.text, file=open('data/players.json','w'))
    
    print ("Downloading competitions data")
    r = requests.get(dataset_links['competitions'], stream=False)
    print (r.text, file=open('data/competitions.json','w'))
    
    print ("Downloading events data")
    r = requests.get(dataset_links['events'], stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("data/events")
    
    print ("Download completed")

def get_statsbomb_data(match_id = 8657):
    with pd.HDFStore(data_folder + "/Statsbomb/data/atomic-spadl-statsbomb.h5",'r') as spadlstore:
        games = (
            spadlstore["games"]
            .merge(spadlstore["competitions"], how='left')
            .merge(spadlstore["teams"].add_prefix('home_'), how='left')
            .merge(spadlstore["teams"].add_prefix('away_'), how='left'))
        
        game = games[(games.game_id == match_id) ]
    #     game = games[(games.competition_name == "FIFA World Cup") 
    #                   & (games.away_team_name == "England")
    #                   & (games.home_team_name == "Belgium")]
        game_id = game.game_id.values[0]
        # print(game_id)
        atomic_actions = spadlstore[f"atomic_actions/game_{game_id}"]
        atomic_actions = (
            atomic_actions
            .merge(spadlstore["atomic_actiontypes"], how="left")
            .merge(spadlstore["bodyparts"], how="left")
            .merge(spadlstore["players"], how="left")
            .merge(spadlstore["teams"], how="left")
        )
        # print(atomic_actions.keys())
    return game, atomic_actions

def plot_goal_actions(match_id = 8657):

    game, atomic_actions = get_statsbomb_data(match_id)
    
    images = []
    for shot in list(atomic_actions[(atomic_actions.type_name == "goal")].index):
        a = atomic_actions[shot-8:shot+1].copy()

        a["start_x"] = a.x
        a["start_y"] = a.y
        a["end_x"] = a.x + a.dx
        a["end_y"] = a.y + a.dy

        g = game.iloc[0]
        minute = int((a.period_id.values[0] - 1) * 45 + a.time_seconds.values[0] // 60)
        game_info = f"{g.game_date} {g.home_team_name} {g.home_score}-{g.away_score} {g.away_team_name} {minute + 1}'"
        # print(game_info)

        def nice_time(row):
            minute = int((row.period_id-1) * 45 + row.time_seconds // 60)
            second = int(row.time_seconds % 60)
            return f"{minute}m{second}s"

        a["nice_time"] = a.apply(nice_time,axis=1)
        labels = a[["nice_time", "type_name", "player_name", "team_name"]]
        
        fig = plt.gcf()
        matplotsoccer.actions(
            location=a[["start_x", "start_y", "end_x", "end_y"]],
            action_type=a.type_name,
            team= a.team_name,
            label=labels,
            labeltitle=["Time", "Action Type", "Player Name", "Team"],
            zoom=False,
            figsize=6,
            show=False,
            color="green"
        )

        imgdata = StringIO()
        plt.savefig(imgdata, format='svg', bbox_inches='tight')
        imgdata.seek(0)

        images.append(imgdata.getvalue())
        plt.cla()

    return images

def pol_heatmap(match_id, playerNames):

    game, atomic_actions = get_statsbomb_data(match_id)
    
    # actions = get_actions(game,data_file)
    # players = pd.read_hdf(data_file,key="players")
    # actions = actions.merge(players)
    print(set(atomic_actions["type_name"]))
    
    atomic_actions = atomic_actions[atomic_actions["type_name"].str.contains("pass|cross|dribble")]
    pa = atomic_actions[atomic_actions["player_name"].str.contains(playerNames)]
    print(pa)

    x = pa.x
    y = pa.y

    fig = plt.gcf()
    # matplotsoccer.field("green",figsize=8,show=False)
    # plt.scatter(x,y)
    # plt.axis("on")
    # matplotsoccer.field("green",figsize=8,show=False)
    # plt.scatter(x,y)
    # plt.show()

    matrix = matplotsoccer.count(x,y,n=50,m=50)
    hm = gaussian_filter(matrix,1)
    hm = matplotsoccer.heatmap(matrix=hm,cmap="RdYlGn_r",linecolor="black")

    # plt.show()
    imgdata = StringIO()
    plt.savefig(imgdata, format='svg', bbox_inches='tight')
    imgdata.seek(0)

    img = imgdata.getvalue()
    return img

def always_ltr(actions):
    away_idx = ~actions.left_to_right
    print(away_idx)
    print(~actions.left_to_right)

    actions.loc[away_idx, "start_x"] = 105 - actions[away_idx].start_x.values
    actions.loc[away_idx, "start_y"] = 68 - actions[away_idx].start_y.values
    actions.loc[away_idx, "end_x"] = 105 - actions[away_idx].end_x.values
    actions.loc[away_idx, "end_y"] = 68 - actions[away_idx].end_y.values
    return actions

def getLineups(competition_id, season_id, match_id):
    lineups = {}
    competitions, matchs, match_events = {}, {}, {}
    with open(data_folder + 'Statsbomb/data/lineups/' + str(match_id) + '.json', encoding="utf-8") as json_data:
        lineups = json.load(json_data)

    with open(data_folder + 'Statsbomb/data/events/' + str(match_id) + '.json', encoding="utf-8") as json_data:
        events = json.load(json_data)
    events = [events[0], events[1]]
    return events

def read_json(path):
    '''
    Read JSON file from path
    '''
    return json.loads(read(path))

def read(path):
    '''
    Read content of a file
    '''
    with open(data_folder + path, 'r', encoding="utf-8") as f:
        return f.read()

def plot_short_event(match_id):
    events = {}
    with open(data_folder + 'Statsbomb/data/events/' + str(match_id) + '.json', encoding="utf-8") as json_data:
    # with open('./data/events/events_%s.json' %tournament) as json_data:
        events = json.load(json_data)

    df = json_normalize(events, sep = "_").assign(match_id = str(match_id))

    #A dataframe of shots
    shots = df.loc[df['type_name'] == 'Shot'].set_index('id')

    pitchLengthX=120
    pitchWidthY=80

    (fig,ax) = createPitch(pitchLengthX,pitchWidthY,'yards','gray')

    home_team_required = events[0]['team']['name']
    away_team_required = events[1]['team']['name']
    #Plot the shots
    for i,shot in shots.iterrows():
        x=shot['location'][0]
        y=shot['location'][1]
        
        goal=shot['shot_outcome_name']=='Goal'
        team_name=shot['team_name']
        
        # circleSize=2
        circleSize=np.sqrt(shot['shot_statsbomb_xg'])*3

        if (team_name==home_team_required):
            if goal:
                shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="red")
                plt.text((x+1),pitchWidthY-y+1,shot['player_name']) 
            else:
                shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="red")     
                shotCircle.set_alpha(.2)
        elif (team_name==away_team_required):
            if goal:
                shotCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="blue") 
                plt.text((pitchLengthX-x+1),y+1,shot['player_name']) 
            else:
                shotCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="blue")      
                shotCircle.set_alpha(.2)
        ax.add_patch(shotCircle)

    plt.text(5,75,away_team_required + ' shots') 
    plt.text(80,75,home_team_required + ' shots') 
        
    fig.set_size_inches(10, 7)
    # fig.savefig('Output/shots.pdf', dpi=100) 
    # plt.show()
    # plt.show()
    imgdata = StringIO()
    plt.savefig(imgdata, format='svg', bbox_inches='tight')
    imgdata.seek(0)

    img = imgdata.getvalue()

    return img

def plot_pass_event(match_id, players):
    events = {}
    with open(data_folder + 'Statsbomb/data/events/' + str(match_id) + '.json', encoding="utf-8") as json_data:
        events = json.load(json_data)

    df = json_normalize(events, sep = "_").assign(match_id = str(match_id))

    #Find the passes
    passes = df.loc[df['type_name'] == 'Pass'].set_index('id')
    #Draw the pitch
    pitchLengthX=120
    pitchWidthY=80
    print(players)
    (fig,ax) = createPitch(pitchLengthX,pitchWidthY,'yards','gray')
    for i,thepass in passes.iterrows():
        if thepass['player_name'] in players:
        # if thepass['team_name']==away_team_required: #
        # if thepass['player_name']=='Marc-André ter Stegen':
            x=thepass['location'][0]
            y=thepass['location'][1]
            passCircle=plt.Circle((x,pitchWidthY-y),1,color="blue")      
            passCircle.set_alpha(.2)   
            ax.add_patch(passCircle)
            dx=thepass['pass_end_location'][0]-x
            dy=thepass['pass_end_location'][1]-y

            passArrow=plt.Arrow(x,pitchWidthY-y,dx,-dy,width=1,color="blue")
            ax.add_patch(passArrow)

    fig.set_size_inches(8, 5)
    # fig.savefig('Output/passes.pdf', dpi=100) 
    # plt.show()
    # plt.show()
    imgdata = StringIO()
    plt.savefig(imgdata, format='svg', bbox_inches='tight')
    imgdata.seek(0)

    img = imgdata.getvalue()

    return img
