from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from .models import Player
from rest_framework import viewsets
from rest_framework import permissions
from .serializers import PlayerSerializer
from pathlib import Path
# import sklearn
import os
import joblib
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize

from .passing_network import passing_network_draw_pitch, draw_pass_map
import matplotlib.pyplot as plt

import plotly.offline as opy
from io import StringIO

from .utils import *
from .plot_utils import *
from .metrics import *
from .FCPython import *
plt.switch_backend('agg')

import warnings
warnings.filterwarnings('ignore')

init_notebook_mode(connected=True)

match_id2match, match_id2events, player_id2player, competition_id2competition, team_id2team = load_public_dataset()
competitions, matchs, match_events = load_Statsbomb_dataset()
# actions = load_statsbomb_data()

def index(request):
    return render(request, 'soccerAnalysisAndPrediction/index.html')

def eva(request):
    return render(request, 'soccerAnalysisAndPrediction/eva.html')

def overviewStatistics(request):
    return render(request, 'soccerAnalysisAndPrediction/overviewStatistics.html')

def matchsCompetitions(request, competition_id, season_id):
    context = {}
    context['competition'] = {}

    data = {}
    for competition in competitions:
        if competition['competition_id'] == competition_id and competition['season_id'] == season_id:
            context['competition'] = competition
            break
    
    # print(matchs[str(16)].keys())
    # print(matchs[str(16)][str(4)])
    context['competition_id'] = competition_id
    context['season_id'] = season_id
    context['matchs'] = matchs[str(competition_id)][str(season_id)] if matchs[str(competition_id)][str(season_id)] != None else []
    
    return render(request, 'soccerAnalysisAndPrediction/matchsCompetitions.html', context)

def matchAnalysis(request, competition_id, season_id, match_id):
    context = {}
    competitionMatchs = matchs[str(competition_id)][str(season_id)]
    match = {}
    for competitionMatch in competitionMatchs:
        if competitionMatch['match_id'] == match_id:
            match = competitionMatch
    

    context['match'] = match
    lineups = getLineups(competition_id, season_id, match_id)
    c = 0
    for p in lineups[0]['tactics']['lineup']:
        lineups[0]['tactics']['lineup'][c]['position']['name'] = ''.join([c for c in p['position']['name'] if c.isupper()])
        c = c + 1

    print(lineups[1])
    c = 0
    for p in lineups[1]['tactics']['lineup']:
        lineups[1]['tactics']['lineup'][c]['position']['name'] = ''.join([c for c in p['position']['name'] if c.isupper()])
        c = c + 1

    context['lineups'] = lineups


    # match_event = match_events[match['match_id']] 
    context['goal_actions'] = plot_goal_actions(match_id)
    # context['heatmap'] = pol_heatmap(match_id)
    context['short_event'] = plot_short_event(match_id)
    context['home_pass_network'] = plot_passing_networks(match_id, match['home_team']['home_team_name'])
    context['away_pass_network'] = plot_passing_networks(match_id, match['away_team']['away_team_name'])

    # context['plot_events'] = plot_events(match_id=match_id)
    # G1, G2 = passing_networks(match_id=match_id)
    # context['plot_passing_networks'] = plot_passing_networks(G1, G2)
    # context['visualize_events'] = visualize_events(match_id=match_id, event_name='all')
    
    return render(request, 'soccerAnalysisAndPrediction/matchAnalysis.html', context)

def pol_MatchHeatmap(request):

    players = list(request.POST.keys())[0]
    # print(request.POST)
    players = json.loads(players)
    print("here", players['match_id'])

    player_names = []
    for player in players:
        if player.find("heatmap_") != -1:
            player_name = player.split("_")[2]
            player_names.append(player_name)
    
    img = pol_heatmap(int(players['match_id']), "|".join(player_names))

    # print(img)

    response = {
        'result' : 'unknow' 
    }
    response['result'] = img
    return JsonResponse(response, safe = False)

def plot_pass_events(request):
    players = list(request.POST.keys())[0]
    # print(request.POST)
    players = json.loads(players)
    print("here", players['match_id'])

    player_names = []
    for player in players:
        if player.find("passEvent_") != -1:
            player_name = player.split("_")[2]
            player_names.append(player_name)
    
    img = plot_pass_event(int(players['match_id']), player_names)

    response = {
        'result' : 'unknow' 
    }
    response['result'] = img
    return JsonResponse(response, safe = False)


def plot_match_events(match_id=2576335, team_id='both', event_name='all'):
    #Size of the pitch in yards (!!!)
    pitchLengthX=120
    pitchWidthY=80

    match_label = match_id2match[match_id]['label']
    match_events = match_id2events[match_id]
    selected_events = []
    for event in match_events:
        if team_id == 'both' or event['teamId'] == team_id:
            if event_name == 'all' or event['eventName'] == event_name:
                selected_events.append(event)
    
    match_df = pd.DataFrame(selected_events)
    match_df['x_start'] = [x[0]['x'] for x in match_df['positions']]
    match_df['y_start'] = [x[0]['y'] for x in match_df['positions']]
    
    if team_id == 'both':
        team_1, team_2 = np.unique(match_df['teamId'])
        df_team_1 = match_df[match_df['teamId'] == team_1]
        df_team_2 = match_df[match_df['teamId'] == team_2]
    else:
        df_team = match_df[match_df['teamId'] == team_id]
    
    f = draw_pitch("white", "black", "h", "full")
    # added
    # fig = plt.figure()

    if team_id == 'both':
        plt.scatter(df_team_1['x_start'], df_team_1['y_start'], c='red', edgecolors="k", zorder=12, 
            alpha=0.5, label='%s: %s %s' %(team_id2team[team_1]['name'], len(df_team_1), 'events' if event_name=='all' else event_name))
        plt.scatter(df_team_2['x_start'], df_team_2['y_start'], marker='s', c='blue', edgecolors="w", linewidth=0.25, zorder=12, 
                    alpha=0.7, label='%s: %s %s' %(team_id2team[team_2]['name'], len(df_team_2), 'events' if event_name=='all' else event_name))
        plt.legend(fontsize=20, bbox_to_anchor=(1.01, 1.05))
    else:
        plt.scatter(df_team['x_start'], df_team['y_start'], 
                    c='red', edgecolors="k", zorder=12, alpha=0.5,
                   label='%s: %s %s' %(team_id2team[team_id]['name'], len(df_team), 'events' if event_name=='all' else event_name))
    plt.title(match_label, fontsize=20)
    plt.legend(fontsize=10, bbox_to_anchor=(1.01, 1.05))
    # plt.show()

    imgdata = StringIO()
    plt.savefig(imgdata, format='svg', bbox_inches='tight')
    imgdata.seek(0)

    data = imgdata.getvalue()
    plt.cla()

    return data

def plot_events(match_id=2576335, team_id='both', event_name='all'):
    """
    Plot the events onthe position where they have been generated.
    
    Parameters
    ----------
    match_id : int, optional
        identifier of the match to plot
        
    team_id : str or int, optional
        the identifier of the team to plot. 
        If 'both', it indicates to plot both teams The default is 'both'.
        
    event_name : str, optional
        the type of the event to plot. If 'all', it plots all the events.
        The defauult is 'all'.
    """
    match_label = match_id2match[match_id]['label']
    match_events = match_id2events[match_id]
    selected_events = []
    for event in match_events:
        if team_id == 'both' or event['teamId'] == team_id:
            if event_name == 'all' or event['eventName'] == event_name:
                selected_events.append(event)
    
    match_df = pd.DataFrame(selected_events)
    match_df['x_start'] = [x[0]['x'] for x in match_df['positions']]
    match_df['y_start'] = [x[0]['y'] for x in match_df['positions']]
    
    if team_id == 'both':
        team_1, team_2 = np.unique(match_df['teamId'])
        df_team_1 = match_df[match_df['teamId'] == team_1]
        df_team_2 = match_df[match_df['teamId'] == team_2]
    else:
        df_team = match_df[match_df['teamId'] == team_id]
    
    f = draw_pitch("white", "black", "h", "full")
    # added
    # fig = plt.figure()

    if team_id == 'both':
        plt.scatter(df_team_1['x_start'], df_team_1['y_start'], c='red', edgecolors="k", zorder=12, 
            alpha=0.5, label='%s: %s %s' %(team_id2team[team_1]['name'], len(df_team_1), 'events' if event_name=='all' else event_name))
        plt.scatter(df_team_2['x_start'], df_team_2['y_start'], marker='s', c='blue', edgecolors="w", linewidth=0.25, zorder=12, 
                    alpha=0.7, label='%s: %s %s' %(team_id2team[team_2]['name'], len(df_team_2), 'events' if event_name=='all' else event_name))
        plt.legend(fontsize=20, bbox_to_anchor=(1.01, 1.05))
    else:
        plt.scatter(df_team['x_start'], df_team['y_start'], 
                    c='red', edgecolors="k", zorder=12, alpha=0.5,
                   label='%s: %s %s' %(team_id2team[team_id]['name'], len(df_team), 'events' if event_name=='all' else event_name))
    plt.title(match_label, fontsize=20)
    plt.legend(fontsize=10, bbox_to_anchor=(1.01, 1.05))
    # plt.show()

    imgdata = StringIO()
    plt.savefig(imgdata, format='svg', bbox_inches='tight')
    imgdata.seek(0)

    data = imgdata.getvalue()
    plt.cla()

    return data

def visualize_events(match_id=2576335, player_id='all', team_id='both', event_name='all'):
    """
    Visualize all the events of a match on the soccer pitch.
    
    Parameters
    ----------
    match_id : int, optional
        identifier of the match to plot
        
    team_id : str or int, optional
        the identifier of the team to plot. 
        If 'both', it indicates to plot both teams The default is 'both'.
        
    event_name : str, optional
        the type of the event to plot. If 'all', it plots all the events.
        The defauult is 'all'.
    """
    
    match_events = []
    for event in match_id2events[match_id]:
        if team_id == 'both' or event['teamId'] == team_id:
            if event_name == 'all' or event['eventName'] == event_name:
                if player_id == 'all' or event['playerId'] == player_id:
                    match_events.append(event)
    
    match = match_id2match[match_id] 
    match_label = match['label']
    
    team1, team2 = match['teamsData'].keys()
    team_name1, team_name2 = team_id2team[int(team1)]['name'], team_id2team[int(team2)]['name']

    # for event in match_events: 
    #     if str(event['teamId']) == team1:
    #         player = player_id2player[event['playerId']]
    #         if player is not None:
    #             print(type(player))
    #             print(player.get('shortName'))

    # print(team_name1)
    
    # Create and style traces
    # text = ['%s by %s (%s)' %(event['eventName'], player_id2player[event['playerId']]['shortName'].encode('ascii', 'strict').decode('unicode-escape'), event['playerId']) for event in match_events if str(event['teamId']) == team1]
    trace1 = go.Scatter(
        x = [event['positions'][0]['x'] for event in match_events if str(event['teamId']) == team1],
        y = [event['positions'][0]['y'] for event in match_events if str(event['teamId']) == team1],
        # text = ['%s by %s (%s)' %(event['eventName'], player_id2player[event['playerId']]['shortName'].encode('ascii', 'strict').decode('unicode-escape'), event['playerId']) for event in match_events if str(event['teamId']) == team1],
        text = ['%s by %s (%s)' %(event['eventName'], str(player_id2player[event['playerId']].get('shortName')).encode('ascii', 'strict').decode('unicode-escape'), event['playerId']) for event in match_events if str(event['teamId']) == team1],
        mode = 'markers',
        name = team_name1,
        marker = dict(
            size = 8,
            color = 'red',
        )
    )

    trace2 = go.Scatter(
        x = [event['positions'][0]['x'] for event in match_events if str(event['teamId']) == team2],
        y = [event['positions'][0]['y'] for event in match_events if str(event['teamId']) == team2],
        # text = ['%s by %s (%s)' %(event['eventName'], player_id2player[event['playerId']]['shortName'].encode('ascii', 'strict').decode('unicode-escape'), event['playerId']) for event in match_events if str(event['teamId']) == team2],
        text = ['%s by %s (%s)' %(event['eventName'], str(player_id2player[event['playerId']].get('shortName')).encode('ascii', 'strict').decode('unicode-escape'), event['playerId']) for event in match_events if str(event['teamId']) == team2],
        mode = 'markers',
        name = team_name2,
        marker = dict(
            size = 8,
            color = 'blue',
            symbol='square'
        )
    )

    fig = dict(data=[trace1, trace2], layout=get_pitch_layout(match_label))
    fig['data'][0]['name'] = team_name1
    fig['data'][1]['name'] = team_name2

    div = opy.plot(fig, auto_open=False, output_type='div')
    # iplot(fig)
    return div

def passing_networks(match_id=2576105):
    """
    Construct the passing networks of the teams in the match.
    
    Parameters
    ----------
    match_id : int, optional
        identifier of the match to plot
        
    Returns
    -------
    tuple
        the two constructed networks, as networkx objects.
    """
    
    # take the names of the two teams of the match
    match_label = match_id2match[match_id]['label']
    team1_name = match_label.split('-')[0].split(' ')[0]
    team2_name = match_label.split('-')[1].split(' ')[1].split(',')[0]
    
    # take all the events of the match
    match_events = []
    for event in match_id2events[match_id]:
        if event['eventName'] == 'Pass':
            match_events.append(event)

    match_events_df = pd.DataFrame(match_events)
    first_half_max_duration = np.max(match_events_df[match_events_df['matchPeriod'] == '1H']['eventSec'])

    # sum 1H time end to all the time in 2H
    for event in match_events:
        if event['matchPeriod'] == '2H':
            event['eventSec'] += first_half_max_duration
    
    team2pass2weight = defaultdict(lambda: defaultdict(int))
    for event, next_event, next_next_event in zip(match_events, match_events[1:], match_events[2:]):
        try:
            if event['eventName'] == 'Pass' and ACCURATE_PASS in [tag['id'] for tag in event['tags']]:
                sender = player_id2player[event['playerId']]['shortName'].encode('ascii', 'strict').decode('unicode-escape')
                # case of duel
                if next_event['eventName'] == 'Duel':
                    # if the next event of from a playero of the same tema
                    if next_event['teamId'] == event['teamId']:
                        receiver = player_id2player[next_event['playerId']]['shortName'].encode('ascii', 'strict').decode('unicode-escape')
                        team2pass2weight[team_id2team[event['teamId']]['name']][(sender, receiver)] += 1
                    else:
                        receiver = player_id2player[next_next_event['playerId']]['shortName'].encode('ascii', 'strict').decode('unicode-escape')
                        team2pass2weight[team_id2team[event['teamId']]['name']][(sender, receiver)] += 1
                else:  # any other event 
                    if next_event['teamId'] == event['teamId']:
                        receiver = player_id2player[next_event['playerId']]['shortName'].encode('ascii', 'strict').decode('unicode-escape')
                        team2pass2weight[team_id2team[event['teamId']]['name']][(sender, receiver)] += 1
        except KeyError:
            pass
    # crete networkx graphs
    G1, G2 = nx.DiGraph(team=team1_name), nx.DiGraph(team=team2_name)
    for (sender, receiver), weight in team2pass2weight[team1_name].items():
        G1.add_edge(sender, receiver, weight=weight)
    for (sender, receiver), weight in team2pass2weight[team2_name].items():
        G2.add_edge(sender, receiver, weight=weight)    
    
    return G1, G2

def plot_passing_networks(match_id = 2576105, team_name = "Russia"):
    passing_networks = []
    data = {}

    lineups_path = "Statsbomb/data/lineups/{0}.json".format(match_id)
    events_path = "Statsbomb/data/events/{0}.json".format(match_id)

    lineups = read_json(lineups_path)
    # print(lineups)

    names_dict = {player["player_name"]: player["player_nickname"]
                for team in lineups for player in team["lineup"]}

    # print(names_dict)

    events = read_json(events_path.format(match_id))
    df_events = json_normalize(events, sep="_").assign(match_id=match_id)

    first_red_card_minute = 90
    if "foul_committed_card_name" in df_events:
        first_red_card_minute = df_events[df_events.foul_committed_card_name.isin(["Second Yellow", "Red Card"])].minute.min()
    # print(first_red_card_minute)
    first_substitution_minute = df_events[df_events.type_name == "Substitution"].minute.min()
    # print(first_red_card_minute)
    max_minute = df_events.minute.max()

    num_minutes = min(first_substitution_minute, first_red_card_minute, max_minute)
    # print(num_minutes)
    
    plot_name = "statsbomb_match{0}_{1}".format(match_id, team_name)

    opponent_team = [x for x in df_events.team_name.unique() if x != team_name][0]
    plot_title ="{0}'s passing network against {1}".format(team_name, opponent_team)

    plot_legend = "Location: pass origin\nSize: number of passes\nColor: number of passes"

    
    df_passes = df_events[(df_events.type_name == "Pass") &
                        (df_events.pass_outcome_name.isna()) &
                        (df_events.team_name == team_name) &
                        (df_events.minute < num_minutes)].copy()

    # If available, use player's nickname instead of full name to optimize space in plot
    df_passes["pass_recipient_name"] = df_passes.pass_recipient_name.apply(lambda x: names_dict[x] if names_dict[x] else x)
    df_passes["player_name"] = df_passes.player_name.apply(lambda x: names_dict[x] if names_dict[x] else x)

    # print(df_passes)

    df_passes["origin_pos_x"] = df_passes.location.apply(lambda x: _statsbomb_to_point(x)[0])
    df_passes["origin_pos_y"] = df_passes.location.apply(lambda x: _statsbomb_to_point(x)[1])
    print(df_passes["origin_pos_x"])
    
    player_position = df_passes.groupby("player_name").agg({"origin_pos_x": "median", "origin_pos_y": "median"})

    player_pass_count = df_passes.groupby("player_name").size().to_frame("num_passes")
    player_pass_value = df_passes.groupby("player_name").size().to_frame("pass_value")

    df_passes["pair_key"] = df_passes.apply(lambda x: "_".join(sorted([x["player_name"], x["pass_recipient_name"]])), axis=1)
    pair_pass_count = df_passes.groupby("pair_key").size().to_frame("num_passes")
    pair_pass_value = df_passes.groupby("pair_key").size().to_frame("pass_value")

    ax = passing_network_draw_pitch()
    ax = draw_pass_map(ax, player_position, player_pass_count, player_pass_value,
                pair_pass_count, pair_pass_value, plot_title, plot_legend)

    imgdata = StringIO()
    plt.savefig(imgdata, format='svg', bbox_inches='tight')
    imgdata.seek(0)
    data = imgdata.getvalue()
    # plt.cla()

    return data

def _statsbomb_to_point(location, max_width=120, max_height=80):
    '''
    Convert a point's coordinates from a StatsBomb's range to 0-1 range.
    '''
    return location[0] / max_width, 1-(location[1] / max_height)

def prediction(request):
    playerData = {}
    all_players = Player.objects.all().values()
    players_object_fields = [f.name for f in Player._meta.get_fields()]
    
    df = pd.DataFrame (all_players)
    # club_names = [club_name for club_name in df['club_name'].unique()]
    # print(df['club_name'].unique())
    club_names = df['club_name'].unique()

    data = {}
    # players = df.copy()
    for index, player in df.iterrows():
        if player['club_name'] not in data:
            data[player['club_name']] = {
                'gk': {},
                'other': {}
            }
        
        # data[player['club_name']] = player
        if player['player_positions'] == "GK":
            data[player['club_name']]["gk"]['#' + str(player['team_jersey_number']) + ' ' + str(player['short_name'])] = str(player['player_id'])
        else:
            data[player['club_name']]["other"]['#' + str(player['team_jersey_number']) + ' ' + str(player['short_name'])] = str(player['player_id'])

        # data[player['club_name']][str(player['team_jersey_number']) + ' ' + str(player['short_name'])] = str(player['player_id'])
    data = json.dumps(data)
    return render(request, 'soccerAnalysisAndPrediction/prediction.html', {'data': data})

def predictResult(request):
    """
    API endpoint that allows users to be viewed or edited.
    """
    players = Player.objects.all().order_by('player_id')
    serializer = PlayerSerializer(players, many = True)

    BASE_DIR = Path(__file__).resolve().parent.parent
    # print(os.path.join(BASE_DIR, 'soccerAnalysisAndPrediction') + '/' + 'prediction_model.sav')
    # os.path.join(BASE_DIR, 'staticfiles')
    loaded_model = joblib.load(os.path.join(BASE_DIR, 'soccerAnalysisAndPrediction') + '/' + 'prediction_model.sav')
    # result = loaded_model.score(X_test, y_test)

    
    attendPlayers = list(request.POST.keys())[0]
    attendPlayers = json.loads(attendPlayers) 

    print(attendPlayers)

    normal_features = [
        "age", "height_cm", "weight_kg", 
        "overall", "potential", "value_eur", "wage_eur",
        "international_reputation", "weak_foot", "skill_moves", 
        "pace", "shooting", "passing", "dribbling", "defending", "physic",
        "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing", "attacking_volleys",
        "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
        "movement_acceleration", "movement_sprint_speed", "movement_agility", "movement_reactions", "movement_balance",
        "power_shot_power", "power_jumping", "power_stamina", "power_strength", "power_long_shots",
        "mentality_aggression", "mentality_interceptions", "mentality_positioning", "mentality_vision", "mentality_composure",
        "defending_standing_tackle", "defending_sliding_tackle",
    ]

    gk_features = [
        "age", "height_cm", "weight_kg", 
        "overall", "potential", "value_eur", "wage_eur",
        "international_reputation", "weak_foot", "skill_moves", 
        "gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning",
        "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning", "goalkeeping_reflexes",
    ]

    floatToIntAtr = [ 
        "pace", "shooting", "passing", "dribbling", "defending", "physic",
        "gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning",
    ]

    formations = [
        'HF_3-1-2-2-2', 'HF_3-1-4-2', 'HF_3-2-2-1-2', 'HF_3-2-2-2-1', 'HF_3-2-3-2', 'HF_3-3-2-1-1', 
        'HF_3-3-2-2', 'HF_3-4-1-2', 'HF_3-4-3', 'HF_3-5-1-1', 'HF_3-5-2', 'HF_4-1-2-3', 'HF_4-1-3-2', 
        'HF_4-1-4-1', 'HF_4-2-2-1-1', 'HF_4-2-2-2', 'HF_4-2-3-1', 'HF_4-3-1-2', 'HF_4-3-2-1', 'HF_4-3-3', 
        'HF_4-4-1-1', 'HF_4-4-2', 'HF_4-5-1', 'HF_5-3-2', 'HF_5-4-1', 
        'AF_3-1-4-1-1', 'AF_3-1-4-2', 'AF_3-2-2-1-2', 'AF_3-2-2-2-1', 'AF_3-2-3-1-1', 'AF_3-2-3-2', 
        'AF_3-3-2-1-1', 'AF_3-3-2-2', 'AF_3-4-1-2', 'AF_3-4-3', 'AF_3-5-1-1', 'AF_3-5-2', 'AF_4-1-3-1-1', 
        'AF_4-1-3-2', 'AF_4-1-4-1', 'AF_4-2-2-1-1', 'AF_4-2-2-2', 'AF_4-2-3-1', 'AF_4-3-1-2', 'AF_4-3-2-1', 
        'AF_4-3-3', 'AF_4-4-1-1', 'AF_4-4-2', 'AF_4-5-1', 'AF_5-3-2', 'AF_5-4-1'
    ]

    data = []

    loopAttrs = ['homeTeam', 'awayTeam']
    for loopAttr in loopAttrs:
        for attPlayer in attendPlayers[loopAttr]:
            team = 'home' if loopAttr == 'homeTeam' else 'away'
            player = Player.objects.filter(player_id=attPlayer['player_id'], club_name= attendPlayers[team])[0]
            player = player.__dict__
            

            features = normal_features if attPlayer['playerPos'] != 'GK' else gk_features
            for feature in features:
                
                val = int(player[feature]) if feature in floatToIntAtr else player[feature]
                data.append(val)
            
            if attPlayer['playerPos'] != 'GK':
                data.append(player[attPlayer['playerPos'].lower()])
            else:
                data.append(player['gk_positioning'])
    
    homeFormation = attendPlayers['homeFormatiom']
    awayFormatiom = attendPlayers['awayFormatiom']
    for formation in formations:
        formation = formation.split('_')
        findFormation = homeFormation if formation[0] == 'HF' else formation[0] == 'AF'

        if formation[1] == findFormation:
            data.append(1)
        else:
            data.append(0)
    
    print(len(data))
    
    prediction = loaded_model.predict([data])
    print(prediction)
    response = {
        'result' : 'unknow' 
    }
    def predictionResult(num):
        return {
                    0: 'Away',
                    1: 'Draw',
                    2: 'Home'
                }[num]
    response['result'] = predictionResult(prediction[0])
    
    return JsonResponse(response, safe = False)

###################################################################
def error_404_view(request, exception):
    return render(request, '404.html')

def error_500_view(request):
    return render(request, '500.html')
