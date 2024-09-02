#!/usr/bin/env python
# coding: utf-8

import fbrefdata as fb
import statsbombpy as st
import mplsoccer as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import json
import os
import shutil
import subprocess
import zipfile
from goalkeeper_comparision import get_stats_lists_gk, read_and_filter_stats_gk

def get_available_leagues():
    """Print available leagues."""
    fb.FBref.available_leagues()

def initialize_fbref(league, season):
    """Initialize FBref object for a given league and season."""
    return fb.FBref(league, season)

def get_stats_lists():
    """Return lists of stats categories."""
    stats_list = ['standard', 'shooting', 'passing', 'passing_types', 'goal_shot_creation', 'defense', 'possession', 'misc']
    keeper_stats_list = ['keeper', 'keeper_adv']
    non_related_list = ['playing_time']
    return stats_list, keeper_stats_list, non_related_list

def read_and_filter_stats_match_player(fbref):
    df_list = []
    df = fbref.read_player_match_stats()
    df_list.append(df)
    return df_list

def read_and_filter_stats(fbref, stats_list):
    """Read and filter player season stats."""
    df_list1 = []
    df_list2 = []
    df1_normal, df1_combined = filter_duplicate_players(fbref.read_player_season_stats('standard'))
    df_list1.append(df1_normal)
    df_list2.append(df1_combined)
    dropped_columns = ['nation', 'pos', 'team', 'age', 'born', 'league', 'season']
    
    for stat in stats_list[1:]:
        df = fbref.read_player_season_stats(stat)
        df.drop(columns=dropped_columns, inplace=True)
        df.fillna(0, inplace=True)
        df_normal, df_combined = filter_duplicate_players(df)
        df_list1.append(df_normal)
        df_list2.append(df_combined)
    
    return df_list1, df_list2

def filter_duplicate_players(df):
    """Filter out players with 2 or more occurrences."""
    player_counts = df['player'].value_counts()
    players_to_drop = player_counts[player_counts >= 2].index

    # Filter the DataFrame to include only players with multiple occurrences
    df_filtered = df[df['player'].isin(players_to_drop)]
    df_normal = df[~df['player'].isin(players_to_drop)]

    # Separate the columns that should not be summed, excluding 'player' and 'id'
    non_sum_columns = df_filtered.select_dtypes(include='object').columns.tolist()
    if 'age' in df_filtered.columns:
        non_sum_columns.append('age')
    if 'born' in df_filtered.columns:
        non_sum_columns.append('born')
    non_sum_columns = list(set(non_sum_columns) - {('player', ''), 'id'})  # Remove duplicates and exclude 'player' and 'id'

    # Adjust non_sum_columns to handle tuples
    res = []
    for x in non_sum_columns:
        if type(x) == tuple and len(x) > 1:
            x = x[0]
        res.append(x)

    # Group by player and id, and calculate the sum of numeric columns excluding non-sum columns
    group_sum = df_filtered.drop(columns=res).groupby(['player', 'id']).sum(numeric_only=True)

    # Concatenate the unique team and pos names for each player and index combination
    try:
        df_filtered['team'] = df_filtered.groupby(['player', df_filtered.index])['team'].transform(lambda x: ', '.join(x.unique()))
    except KeyError:
        pass

    try:
        df_filtered['pos'] = df_filtered.groupby(['player', df_filtered.index])['pos'].transform(lambda x: ', '.join(set(', '.join(x).split(', '))))
    except KeyError:
        pass

    # Handle non-sum columns by taking the first occurrence
    group_non_sum = df_filtered[['player'] + res]

    # Reset index for merging
    group_sum = group_sum.reset_index().set_index('id')
    group_combined = group_sum.reset_index().merge(group_non_sum, on=['player', 'id'])

    # Drop duplicates
    group_combined = group_combined.drop_duplicates(keep='first')
    group_combined = group_combined.set_index('id')
    # print(group_combined)
    # combined_df = pd.concat([df_normal, group_combined]).drop_duplicates(keep='last')
    return df_normal, group_combined

def merge_dataframes(df_list1, df_list2):
    """Merge two lists of dataframes on the 'player' column."""
    for i in range(len(df_list1)):
        df_list1[i] = df_list1[i].reset_index()
        df_list2[i] = df_list2[i].reset_index()
    
    merged_df1 = df_list1[0]
    merged_df2 = df_list2[0]
    
    for i, (df1, df2) in enumerate(zip(df_list1[1:], df_list2[1:]), start=1):
        merged_df1 = pd.merge(merged_df1, df1, on='player', how='inner', suffixes=('', f'_df{i}'))
        merged_df2 = pd.merge(merged_df2, df2, on='player', how='inner', suffixes=('', f'_df{i}'))
    
    merged_df1.set_index('id', inplace=True)
    merged_df2.set_index('id', inplace=True)
    
    return merged_df1, merged_df2

def get_player_values(merged_df1, merged_df2, player_name, col):
    """Get player values for radar chart from two merged DataFrames."""
    player_df1 = merged_df1[merged_df1['player'] == player_name]
    player_df2 = merged_df2[merged_df2['player'] == player_name]
    
    if not player_df1.empty:
        player_df = player_df1
    else:
        player_df = player_df2
    
    player_values = np.array(player_df[col[0][0]][col[0][1]].values[0])
    
    for x in col[1:]:
        if len(x) == 2:
            player_values = np.append(player_values, player_df[x[0]][x[1]].values[0])
        else:
            player_values = np.append(player_values, player_df[x[0]].values[0])
    
    return player_values

def create_radar_chart(params, low, high, lower_is_better, player1_values, player2_values, player1_name, player2_name, team1_name, team2_name):
    """Create radar chart comparing two players."""
    radar = mp.Radar(params, low, high, lower_is_better=lower_is_better, round_int=[False]*len(params), num_rings=4, ring_width=1, center_circle_radius=1)
    
    URL1 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/SourceSerifPro-Regular.ttf')
    serif_regular = mp.FontManager(URL1)
    URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/SourceSerifPro-ExtraLight.ttf')
    serif_extra_light = mp.FontManager(URL2)
    URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/RubikMonoOne-Regular.ttf')
    rubik_regular = mp.FontManager(URL3)
    URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
    robotto_thin = mp.FontManager(URL4)
    URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf')
    robotto_bold = mp.FontManager(URL5)
    
    fig, axs = mp.grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025, title_space=0, endnote_space=0, grid_key='radar', axis=False)
    
    radar.setup_axis(ax=axs['radar'])
    rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='#ffb2b2', edgecolor='#fc5f5f')
    radar_output = radar.draw_radar_compare(player1_values, player2_values, ax=axs['radar'], kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6}, kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
    radar_poly, radar_poly2, vertices1, vertices2 = radar_output
    range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=25, fontproperties=robotto_thin.prop)
    param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=25, fontproperties=robotto_thin.prop)
    axs['radar'].scatter(vertices1[:, 0], vertices1[:, 1], c='#00f2c1', edgecolors='#6d6c6d', marker='o', s=150, zorder=2)
    axs['radar'].scatter(vertices2[:, 0], vertices2[:, 1], c='#d80499', edgecolors='#6d6c6d', marker='o', s=150, zorder=2)
    
    endnote_text = axs['endnote'].text(0.99, 0.5, 'Inspired By: StatsBomb / Rami Moghadam', fontsize=15, fontproperties=robotto_thin.prop, ha='right', va='center')
    title1_text = axs['title'].text(0.01, 0.65, player1_name, fontsize=25, color='#01c49d', fontproperties=robotto_bold.prop, ha='left', va='center')
    title2_text = axs['title'].text(0.01, 0.25, team1_name, fontsize=20, fontproperties=robotto_thin.prop, ha='left', va='center', color='#01c49d')
    title3_text = axs['title'].text(0.99, 0.65, player2_name, fontsize=25, fontproperties=robotto_bold.prop, ha='right', va='center', color='#d80499')
    title4_text = axs['title'].text(0.99, 0.25, team2_name, fontsize=20, fontproperties=robotto_thin.prop, ha='right', va='center', color='#d80499')
    
    fig.set_facecolor('#f2dad2')
    st.pyplot(fig)

def compare_players_and_create_radar(merged_df1, merged_df2, player1, player2, selected_params, param_mapping, lower_is_better):
    """Compare two players and create a radar chart."""
    col = [param_mapping[param] for param in selected_params]
    player1_values = get_player_values(merged_df1, merged_df2, player1, col)
    player2_values = get_player_values(merged_df1, merged_df2, player2, col)
    
    # Get team names
    if player1 in merged_df1['player'].values:
        team1_name = merged_df1[merged_df1['player'] == player1]['team'].values[0]
    else:
        team1_name = merged_df2[merged_df2['player'] == player1]['team'].values[0]
    
    if player2 in merged_df1['player'].values:
        team2_name = merged_df1[merged_df1['player'] == player2]['team'].values[0]
    else:
        team2_name = merged_df2[merged_df2['player'] == player2]['team'].values[0]
    
    # Define lower and upper limits for each parameter
    predefined_low = [0] * len(selected_params)
    predefined_high = [1] * len(selected_params)
    
    low = np.minimum(predefined_low, np.minimum(player1_values, player2_values))
    high = np.maximum(predefined_high, np.maximum(player1_values, player2_values))
    
    # Ensure that high is greater than low for all parameters
    for i in range(len(low)):
        if low[i] >= high[i]:
            high[i] = low[i] + 1  # Adjust high to be greater than low
    
    create_radar_chart(selected_params, low, high, lower_is_better, player1_values, player2_values, player1, player2, team1_name, team2_name)

def create_radar_chart_match_compare(params, low, high, lower_is_better, player1_values, player2_values, player1_name, player2_name, team1_name, team2_name, match1_name, match2_name):
    """Create radar chart comparing two players."""
    radar = mp.Radar(params, low, high, lower_is_better=lower_is_better, round_int=[False]*len(params), num_rings=4, ring_width=1, center_circle_radius=1)
    
    URL1 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/SourceSerifPro-Regular.ttf')
    serif_regular = mp.FontManager(URL1)
    URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/SourceSerifPro-ExtraLight.ttf')
    serif_extra_light = mp.FontManager(URL2)
    URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/RubikMonoOne-Regular.ttf')
    rubik_regular = mp.FontManager(URL3)
    URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
    robotto_thin = mp.FontManager(URL4)
    URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf')
    robotto_bold = mp.FontManager(URL5)
    
    fig, axs = mp.grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025, title_space=0, endnote_space=0, grid_key='radar', axis=False)
    
    radar.setup_axis(ax=axs['radar'])
    rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='#ffb2b2', edgecolor='#fc5f5f')
    radar_output = radar.draw_radar_compare(player1_values, player2_values, ax=axs['radar'], kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6}, kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
    radar_poly, radar_poly2, vertices1, vertices2 = radar_output
    range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=25, fontproperties=robotto_thin.prop)
    param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=25, fontproperties=robotto_thin.prop)
    axs['radar'].scatter(vertices1[:, 0], vertices1[:, 1], c='#00f2c1', edgecolors='#6d6c6d', marker='o', s=150, zorder=2)
    axs['radar'].scatter(vertices2[:, 0], vertices2[:, 1], c='#d80499', edgecolors='#6d6c6d', marker='o', s=150, zorder=2)
    
    endnote_text = axs['endnote'].text(0.99, 0.5, 'Inspired By: StatsBomb / Rami Moghadam', fontsize=15, fontproperties=robotto_thin.prop, ha='right', va='center')
    title1_text = axs['title'].text(0.01, 0.65, player1_name, fontsize=25, color='#01c49d', fontproperties=robotto_bold.prop, ha='left', va='center')
    title2_text = axs['title'].text(0.01, 0.25, team1_name, fontsize=20, fontproperties=robotto_thin.prop, ha='left', va='center', color='#01c49d')
    title3_text = axs['title'].text(0.99, 0.65, player2_name, fontsize=25, fontproperties=robotto_bold.prop, ha='right', va='center', color='#d80499')
    title4_text = axs['title'].text(0.99, 0.25, team2_name, fontsize=20, fontproperties=robotto_thin.prop, ha='right', va='center', color='#d80499')
    title5_text = axs['title'].text(0.01, -0.2, match1_name, fontsize=20, fontproperties=robotto_thin.prop, ha='left', va='center', color='#01c49d')
    title6_text = axs['title'].text(0.99, -0.2, match2_name, fontsize=20, fontproperties=robotto_thin.prop, ha='right', va='center', color='#d80499')
    
    fig.set_facecolor('#f2dad2')
    st.pyplot(fig)

def compare_match_players_and_create_radar(merged_df, player1, player2, match1_name, match2_name, selected_params, param_mapping, lower_is_better):
    """Compare two players and create a radar chart."""
    col = [param_mapping[param] for param in selected_params]
    player1_values = get_player_match_values(merged_df, player1, match1_name, col)
    player2_values = get_player_match_values(merged_df, player2, match2_name, col)
    
    # Get team names
    team1_name = merged_df[merged_df['player'] == player1]['team'].values[0]
    team2_name = merged_df[merged_df['player'] == player2]['team'].values[0]
    
    # Define lower and upper limits for each parameter
    predefined_low = [0] * len(selected_params)
    predefined_high = [1] * len(selected_params)
    
    low = np.minimum(predefined_low, np.minimum(player1_values, player2_values))
    high = np.maximum(predefined_high, np.maximum(player1_values, player2_values))
    
    # Ensure that high is greater than low for all parameters
    for i in range(len(low)):
        if low[i] >= high[i]:
            high[i] = low[i] + 1  # Adjust high to be greater than low
    
    create_radar_chart_match_compare(selected_params, low, high, lower_is_better, player1_values, player2_values, player1, player2, team1_name, team2_name, match1_name, match2_name)

def get_player_match_values(merged_df, player_name, match_name, col):
    """Get player values for radar chart."""
    player_df = merged_df[(merged_df['player'] == player_name) & (merged_df['game'] == match_name)]
    player_values = np.array(player_df[col[0][0]][col[0][1]].values[0])
    
    for x in col[1:]:
        if len(x) == 2:
            player_values = np.append(player_values, player_df[x[0]][x[1]].values[0])
        else:
            player_values = np.append(player_values, player_df[x[0]].values[0])
    
    return player_values

def player_match_compare():
    with open('league_dict.json', 'r') as f:
        league_dict = json.load(f)

    # Create a mapping from league names to keys
    league_name_to_key = {v['FBref']: k for k, v in league_dict.items()}

    # User selects the league
    league_options = list(league_name_to_key.keys())
    selected_league_name = st.selectbox("Select the league", league_options, key='match_league')

    # Get the corresponding league key
    selected_league_key = league_name_to_key[selected_league_name]

    # User selects the season
    season = st.selectbox("Select the season", ['2024-2025', '2023-2024', '2022-2023', '2021-2022', '2020-2021', '2019-2020', '2018-2019', '2017-2018'], key='match_season')

    if 'match_data_loaded' not in st.session_state or st.session_state['selected_league_name'] != selected_league_name or st.session_state['season'] != season:
        fbref = initialize_fbref(selected_league_key, season)
        stats_list, _, _ = get_stats_lists()
        df_list = read_and_filter_stats_match_player(fbref)

        st.session_state['match_data_loaded'] = True
        st.session_state['merged_df'] = df_list[0]
        st.session_state['selected_league_name'] = selected_league_name
        st.session_state['season'] = season

    merged_df = st.session_state['merged_df']

    param_mapping = {
        "Goals": ['Performance', 'Gls'],
        "Assists": ['Performance', 'Ast'],
        "Penalty Goals": ['Performance', 'PK'],
        "Penalty Attempts": ['Performance', 'PKatt'],
        "Shots": ['Performance', 'Sh'],
        "Shots on Target": ['Performance', 'SoT'],
        "Yellow Cards": ['Performance', 'CrdY'],
        "Red Cards": ['Performance', 'CrdR'],
        "Touches": ['Performance', 'Touches'],
        "Tackles": ['Performance', 'Tkl'],
        "Interceptions": ['Performance', 'Int'],
        "Blocks": ['Performance', 'Blocks'],
        "xG": ['Expected', 'xG'],
        "npxG": ['Expected', 'npxG'],
        "xAG": ['Expected', 'xAG'],
        "Shot-Creating Actions": ['SCA', 'SCA'],
        "Goal-Creating Actions": ['SCA', 'GCA'],
        "Passes Completed": ['Passes', 'Cmp'],
        "Passes Attempted": ['Passes', 'Att'],
        "Pass Completion %": ['Passes', 'Cmp%'],
        "Progressive Passes": ['Passes', 'PrgP'],
        "Carries": ['Carries', 'Carries'],
        "Progressive Carries": ['Carries', 'PrgC'],
        "Take-Ons Attempted": ['Take-Ons', 'Att'],
        "Successful Take-Ons": ['Take-Ons', 'Succ']
    }

    # Get list of players
    players = merged_df['player'].unique().tolist()

    # Get list of games
    games = merged_df['game'].unique().tolist()

    # Streamlit UI elements
    player1 = st.selectbox("Select the first player", players)
    player2 = st.selectbox("Select the second player", players)

    selected_game1 = st.selectbox("Select the match for the first player", games)
    selected_game2 = st.selectbox("Select the match for the second player", games)

    params = list(param_mapping.keys())
    selected_params = st.multiselect("Select parameters to compare (make sure to choose 3 or more parameters)", params, default=params[:5])

    lower_is_better_options = st.multiselect("Select parameters where lower is better", params)

    if st.button("Compare Players"):
        compare_match_players_and_create_radar(merged_df, player1, player2, selected_game1, selected_game2, selected_params, param_mapping, lower_is_better_options)



def player_season_compare():
    with open('league_dict.json', 'r') as f:
        league_dict = json.load(f)
    
    # Create a mapping from league names to keys
    league_name_to_key = {v['FBref']: k for k, v in league_dict.items()}
    
    # User selects the league
    league_options = list(league_name_to_key.keys())
    selected_league_name = st.selectbox("Select the league", league_options)
    
    # Get the corresponding league key
    selected_league_key = league_name_to_key[selected_league_name]
    
    # User selects the season
    season = st.selectbox("Select the season", ['2024-2025', '2023-2024', '2022-2023', '2021-2022', '2020-2021', '2019-2020', '2018-2019', '2017-2018'])
    fbref = initialize_fbref(selected_league_key, season)

    # User selects the type of comparison
    comparison_type = st.radio("Choose comparison type", ('Outfielder', 'Goalkeeper'))

    if comparison_type == 'Outfielder':
        stats_list, _, _ = get_stats_lists()
        df_list1, df_list2 = read_and_filter_stats(fbref, stats_list)
        param_mapping = {
            "Goals": ['Performance', 'Gls'],
            "Assists": ['Performance', 'Ast'],
            "Goals + Assists": ['Performance', 'G+A'],
            "Non-Penalty Goals": ['Performance', 'G-PK'],
            "Penalty Goals": ['Performance', 'PK'],
            'Penalty kick Attempts': ['Performance', 'PKatt'],
            "Yellow Cards": ['Performance', 'CrdY'],
            "Red Cards": ['Performance', 'CrdR'],

            "xG": ['Expected', 'xG'],
            "npxG": ['Expected', 'npxG'],
            "xAG": ['Expected', 'xAG'],
            "npxG+xAG": ['Expected', 'npxG+xAG'],

            "Progressive Carries": ['Progression', 'PrgC'],
            "Progressive Pass": ['Progression', 'PrgP'],
            "Progressive Pass Received": ['Progression', 'PrgR'],
            "Goals/90": ["Per 90 Minutes", "Gls"],
            "Assists/90": ["Per 90 Minutes", "Ast"],
            "Goals+Assists/90": ["Per 90 Minutes", "G+A"],
            "Non-Penalty Goals/90": ["Per 90 Minutes", "G-PK"],
            "Non-Penalty Goals + Assists/90": ["Per 90 Minutes", "G+A-PK"],
            "xG/90": ["Per 90 Minutes", "xG"],
            "xAG/90": ["Per 90 Minutes", "xAG"],
            "xG+xAG/90": ["Per 90 Minutes", "xG+xAG"],
            "npxG/90": ["Per 90 Minutes", "npxG"],
            "npxG+xAG/90": ["Per 90 Minutes", "npxG+xAG"],

            "Key Passes": ['KP'],
            "Through Balls": ['Pass Types', 'TB'],
            "Shot-Creating Actions": ['SCA', 'SCA'],
            "Goal-Creating Actions": ['GCA', 'GCA'],
            "Touches In Attacking 1/3": ['Touches', 'Att 3rd'],
            "Miscontrol": ['Carries', 'Mis'],
            "Dispossessed": ['Carries', 'Dis'],

            "Shots": ['Standard', 'Sh'],
            "Shots on Target": ['Standard', 'SoT'],
            "Shots on Target %": ['Standard', 'SoT%'],
            "Shots/90": ['Standard', 'Sh/90'],
            "Shots on Target/90": ['Standard', 'SoT/90'],
            "Goals per Shot": ['Standard', 'G/Sh'],
            "Goals per Shot on Target": ['Standard', 'G/SoT'],
            "Shot Distance": ['Standard', 'Dist'],
            "Free Kick Goals": ['Standard', 'FK'],
            "Penalty Attempts": ['Standard', 'PKatt'],

            "Total Passes Completed": ['Total', 'Cmp'],
            "Total Passes Attempted": ['Total', 'Att'],
            "Total Pass Completion %": ['Total', 'Cmp%'],
            "Total Pass Distance": ['Total', 'TotDist'],
            "Progressive Pass Distance": ['Total', 'PrgDist'],

            "Short Passes Completed": ['Short', 'Cmp'],
            "Short Passes Attempted": ['Short', 'Att'],
            "Short Pass Completion %": ['Short', 'Cmp%'],

            "Medium Passes Completed": ['Medium', 'Cmp'],
            "Medium Passes Attempted": ['Medium', 'Att'],
            "Medium Pass Completion %": ['Medium', 'Cmp%'],

            "Long Passes Completed": ['Long', 'Cmp'],
            "Long Passes Attempted": ['Long', 'Att'],
            "Long Pass Completion %": ['Long', 'Cmp%'],

            "Expected Assists": ['Expected', 'xA'],
            "Assists minus xAG": ['Expected', 'A-xAG'],

            "Passes into Final Third": ['1/3'],
            "Passes into Penalty Area": ['PPA'],
            "Crosses into Penalty Area": ['CrsPA'],

            "Live Passes": ['Pass Types', 'Live'],
            "Dead Passes": ['Pass Types', 'Dead'],
            "Free Kick Passes": ['Pass Types', 'FK'],
            "Switches": ['Pass Types', 'Sw'],
            "Crosses": ['Pass Types', 'Crs'],
            "Throw-ins": ['Pass Types', 'TI'],
            "Corner Kicks": ['Pass Types', 'CK'],

            "In-swinging Corners": ['Corner Kicks', 'In'],
            "Out-swinging Corners": ['Corner Kicks', 'Out'],
            "Straight Corners": ['Corner Kicks', 'Str'],

            "Passes Completed": ['Outcomes', 'Cmp'],
            "Passes Offside": ['Outcomes', 'Off'],
            "Passes Blocked": ['Outcomes', 'Blocks'],

            "Shot-Creating Actions per 90": ['SCA', 'SCA90'],
            "SCA from Live Passes": ['SCA Types', 'PassLive'],
            "SCA from Dead Passes": ['SCA Types', 'PassDead'],
            "SCA from Take-Ons": ['SCA Types', 'TO'],
            "SCA from Shots": ['SCA Types', 'Sh'],
            "SCA from Fouls Drawn": ['SCA Types', 'Fld'],
            "SCA from Defensive Actions": ['SCA Types', 'Def'],

            "Goal-Creating Actions per 90": ['GCA', 'GCA90'],
            "GCA from Live Passes": ['GCA Types', 'PassLive'],
            "GCA from Dead Passes": ['GCA Types', 'PassDead'],
            "GCA from Take-Ons": ['GCA Types', 'TO'],
            "GCA from Shots": ['GCA Types', 'Sh'],
            "GCA from Fouls Drawn": ['GCA Types', 'Fld'],
            "GCA from Defensive Actions": ['GCA Types', 'Def'],

            "Tackles": ['Tackles', 'Tkl'],
            "Tackles Won": ['Tackles', 'TklW'],
            "Tackles in Defensive Third": ['Tackles', 'Def 3rd'],
            "Tackles in Midfield Third": ['Tackles', 'Mid 3rd'],
            "Tackles in Attacking Third": ['Tackles', 'Att 3rd'],

            "Challenges": ['Challenges', 'Tkl'],
            "Challenges Attempted": ['Challenges', 'Att'],
            "Challenge Success %": ['Challenges', 'Tkl%'],
            "Challenges Lost": ['Challenges', 'Lost'],

            "Blocks": ['Blocks', 'Blocks'],
            "Shots Blocked": ['Blocks', 'Sh'],
            "Passes Blocked": ['Blocks', 'Pass'],

            "Interceptions": ['Int'],
            "Tackles + Interceptions": ['Tkl+Int'],
            "Clearances": ['Clr'],
            "Errors": ['Err'],

            "Touches": ['Touches', 'Touches'],
            "Touches in Defensive Penalty Area": ['Touches', 'Def Pen'],
            "Touches in Defensive Third": ['Touches', 'Def 3rd'],
            "Touches in Midfield Third": ['Touches', 'Mid 3rd'],
            "Touches in Attacking Third": ['Touches', 'Att 3rd'],
            "Touches in Attacking Penalty Area": ['Touches', 'Att Pen'],
            "Live-Ball Touches": ['Touches', 'Live'],

            "Take-Ons Attempted": ['Take-Ons', 'Att'],
            "Successful Take-Ons": ['Take-Ons', 'Succ'],
            "Successful Take-On %": ['Take-Ons', 'Succ%'],
            "Take-Ons Tkl": ['Take-Ons', 'Tkld'],
            "Take-Ons Tkl %": ['Take-Ons', 'Tkld%'],

            "Carries": ['Carries', 'Carries'],
            "Total Carry Distance": ['Carries', 'TotDist'],
            "Progressive Carry Distance": ['Carries', 'PrgDist'],
            "Carries into Final Third": ['Carries', '1/3'],
            "Carries into Penalty Area": ['Carries', 'CPA'],

            "Passes Received": ['Receiving', 'Rec'],
            "Progressive Passes Received": ['Receiving', 'PrgR'],

            "Penalty Kicks Won": ['Performance_df7', 'PKwon'],
            "Penalty Kicks Conceded": ['Performance_df7', 'PKcon'],
            "Own Goals": ['Performance_df7', 'OG'],
            "Recoveries": ['Performance_df7', 'Recov'],

            "Aerial Duels Won": ['Aerial Duels', 'Won'],
            "Aerial Duels Lost": ['Aerial Duels', 'Lost'],
            "Aerial Duels Won %": ['Aerial Duels', 'Won%']
        }
    else:
        stats_list, _, _ = get_stats_lists_gk()
        df_list1, df_list2 = read_and_filter_stats_gk(fbref, stats_list)
        param_mapping = {
            "Goals Against": ['Performance', 'GA'],
            "Goals Against per 90": ['Performance', 'GA90'],
            "Shots on Target Against": ['Performance', 'SoTA'],
            "Saves": ['Performance', 'Saves'],
            "Save Percentage": ['Performance', 'Save%'],
            "Clean Sheets": ['Performance', 'CS'],
            "Clean Sheet Percentage": ['Performance', 'CS%'],
            "Penalty Kicks Allowed": ['Penalty Kicks', 'PKA'],
            "Penalty Kicks Saved": ['Penalty Kicks', 'PKsv'],
            "Penalty Save Percentage": ['Penalty Kicks', 'Save%'],
            "Post-Shot Expected Goals": ['Expected', 'PSxG'],
            "Post-Shot Expected Goals per Shot on Target": ['Expected', 'PSxG/SoT'],
            "Post-Shot Expected Goals +/-": ['Expected', 'PSxG+/-'],
            "Post-Shot Expected Goals per 90": ['Expected', '/90'],
            "Launched Passes Completed": ['Launched', 'Cmp'],
            "Launched Passes Attempted": ['Launched', 'Att'],
            "Launched Pass Completion Percentage": ['Launched', 'Cmp%'],
            "Passes Attempted (GK)": ['Passes', 'Att (GK)'],
            "Throws": ['Passes', 'Thr'],
            "Launch Percentage": ['Passes', 'Launch%'],
            "Average Pass Length": ['Passes', 'AvgLen'],
            "Goal Kicks Attempted": ['Goal Kicks', 'Att'],
            "Goal Kicks Launch Percentage": ['Goal Kicks', 'Launch%'],
            "Average Goal Kick Length": ['Goal Kicks', 'AvgLen'],
            "Crosses Faced": ['Crosses', 'Opp'],
            "Crosses Stopped": ['Crosses', 'Stp'],
            "Crosses Stopped Percentage": ['Crosses', 'Stp%'],
            "Sweeper Actions": ['Sweeper', '#OPA'],
            "Sweeper Actions per 90": ['Sweeper', '#OPA/90'],
            "Average Sweeper Action Distance": ['Sweeper', 'AvgDist'],
            "Minutes Played": ['Playing Time', 'Min'],
            "Matches Played": ['Playing Time', 'MP'],
            "Progressive Carries": ['Progression', 'PrgC'],
            "Progressive Passes": ['Progression', 'PrgP'],
            "Progressive Passes Received": ['Progression', 'PrgR'],
            "Passes Completed": ['Total', 'Cmp'],
            "Passes Attempted": ['Total', 'Att'],
            "Pass Completion Percentage": ['Total', 'Cmp%'],
            "Total Passing Distance": ['Total', 'TotDist'],
            "Progressive Passing Distance": ['Total', 'PrgDist'],
            "Short Passes Completed": ['Short', 'Cmp'],
            "Short Passes Attempted": ['Short', 'Att'],
            "Short Pass Completion Percentage": ['Short', 'Cmp%'],
            "Medium Passes Completed": ['Medium', 'Cmp'],
            "Medium Passes Attempted": ['Medium', 'Att'],
            "Medium Pass Completion Percentage": ['Medium', 'Cmp%'],
            "Long Passes Completed": ['Long', 'Cmp'],
            "Long Passes Attempted": ['Long', 'Att'],
            "Long Pass Completion Percentage": ['Long', 'Cmp%'],
            "Key Passes": ['KP'],
            "Passes into Final Third": ['1/3'],
            "Passes into Penalty Area": ['PPA'],
            "Shot-Creating Actions": ['SCA', 'SCA'],
            "Shot-Creating Actions per 90": ['SCA', 'SCA90'],
            "Goal-Creating Actions": ['GCA', 'GCA'],
            "Goal-Creating Actions per 90": ['GCA', 'GCA90'],
            "Tackles": ['Tackles', 'Tkl'],
            "Tackles Won": ['Tackles', 'TklW'],
            "Tackles in Defensive Third": ['Tackles', 'Def 3rd'],
            "Challenges": ['Challenges', 'Tkl'],
            "Challenges Attempted": ['Challenges', 'Att'],
            "Challenge Success Percentage": ['Challenges', 'Tkl%'],
            "Challenges Lost": ['Challenges', 'Lost'],
            "Passes Blocked": ['Blocks', 'Pass'],
            "Interceptions": ['Int'],
            "Tackles + Interceptions": ['Tkl+Int'],
            "Clearances": ['Clr'],
            "Errors": ['Err'],
            "Touches": ['Touches', 'Touches'],
            "Touches in Defensive Penalty Area": ['Touches', 'Def Pen'],
            "Touches in Defensive Third": ['Touches', 'Def 3rd'],
            "Touches in Midfield Third": ['Touches', 'Mid 3rd'],
            "Live-Ball Touches": ['Touches', 'Live'],
            "Carries": ['Carries', 'Carries'],
            "Total Carrying Distance": ['Carries', 'TotDist'],
            "Progressive Carrying Distance": ['Carries', 'PrgDist'],
            "Progressive Carries": ['Carries', 'PrgC'],
            "Miscontrols": ['Carries', 'Mis'],
            "Dispossessed": ['Carries', 'Dis'],
            "Passes Received": ['Receiving', 'Rec'],
            "Progressive Passes Received": ['Receiving', 'PrgR'],
            "Fouls Committed": ['Performance_df7', 'Fls'],
            "Fouls Drawn": ['Performance_df7', 'Fld'],
            "Interceptions": ['Performance_df7', 'Int'],
            "Penalty Kicks Conceded": ['Performance_df7', 'PKcon'],
            "Own Goals": ['Performance_df7', 'OG'],
            "Recoveries": ['Performance_df7', 'Recov'],
            "Aerial Duels Won": ['Aerial Duels', 'Won'],
            "Aerial Duels Lost": ['Aerial Duels', 'Lost'],
            "Aerial Duels Win Percentage": ['Aerial Duels', 'Won%']
        }
    merged_df1, merged_df2 = merge_dataframes(df_list1, df_list2)
    # Get list of players
    players = merged_df1['player'].unique().tolist()
    players.extend(merged_df2['player'].unique().tolist())
    players = list(set(players))  # Remove duplicates
    
    player1 = st.selectbox("Select the first player", players)
    player2 = st.selectbox("Select the second player", players)
    
    params = list(param_mapping.keys())
    selected_params = st.multiselect("Select parameters to compare (make sure to choose 3 or more parameters)", params, default=params[:5])
    
    lower_is_better_options = st.multiselect("Select parameters where lower is better", params)
    
    if st.button("Compare Players"):
        compare_players_and_create_radar(merged_df1, merged_df2, player1, player2, selected_params, param_mapping, lower_is_better_options)

# Function to copy the folder
def copy_folder(source_dir, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy the entire directory and replace files with the same name
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        dest_item = os.path.join(dest_dir, item)
        
        if os.path.isdir(source_item):
            if os.path.exists(dest_item):
                shutil.rmtree(dest_item, ignore_errors=True)
            shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(source_item, dest_item)

# Function to delete specific files
def delete_specific_files(directory, pattern):
    for item in os.listdir(directory):
        if pattern in item:
            file_path = os.path.join(directory, item)
            if os.path.isfile(file_path):
                os.remove(file_path)
                


# Function to perform git operations
def git_operations():
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', 'Auto commit from Streamlit app'])
    subprocess.run(['git', 'push'])

# Check if the folder has already been copied

def copy_league_dict_json():
    fbrefdata_dir = os.getenv('FBREFDATA_DIR', os.path.expanduser('~/fbrefdata'))
    config_dir = os.path.join(fbrefdata_dir, 'config')
    league_dict_path = os.path.join(config_dir, 'league_dict.json')
    # Ensure the directory exists
    os.makedirs(config_dir, exist_ok=True)
    source = 'league_dict.json'
    shutil.copy(source, league_dict_path)

# Function to zip the data folder
def zip_data_folder(source_dir, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, source_dir))

# Function to provide download link for the zip file
def provide_download_link(zip_name):
    with open(zip_name, 'rb') as f:
        st.download_button(
            label="Download Data Folder",
            data=f,
            file_name=zip_name,
            mime='application/zip'
        )

def check_and_copy_folder():
    flag_file = 'folder_copied.flag'
    fbrefdata_dir = os.getenv('FBREFDATA_DIR', os.path.expanduser('~/fbrefdata/data'))
    source_dir = os.path.join(os.getcwd(), 'fbrefdata/data')
    
    if not os.path.exists(flag_file):
        copy_folder(source_dir, fbrefdata_dir)
        copy_league_dict_json()
        with open(flag_file, 'w') as f:
            f.write('Folder copied')
        st.session_state['folder_copied'] = True
    else:
        # Reverse copy after refresh
        copy_folder(fbrefdata_dir, source_dir)
        git_operations()
        delete_specific_files(source_dir, 'teams')
        delete_specific_files(fbrefdata_dir, 'teams')
        os.remove(flag_file)
        st.session_state['folder_copied'] = False
    
    # Zip the data folder and provide download link
    zip_name = 'data_folder.zip'
    zip_data_folder(source_dir, zip_name)
    provide_download_link(zip_name)

# Initialize session state
if 'folder_copied' not in st.session_state:
    check_and_copy_folder()
# Function to load data
def load_data():
    # Simulate data loading
    st.session_state['data_loaded'] = True
    st.session_state['data'] = "Your data here"

# Initialize session state for data loading
if 'data_loaded' not in st.session_state:
    load_data()
def main():

    
    st.title("Comparison Radar Chart")
    # 
    # Load league dictionary
    
    # User selects the type of comparison
    comparison_choice = st.radio("Choose comparison type", ('Player in Season', 'Player in Matches'))

    if comparison_choice == 'Player in Season':
        # Call the function for comparing players in matches
        player_season_compare()
        provide_download_link()
    elif comparison_choice == 'Player in Matches':
        # Call the function for comparing players in season
        player_match_compare()
        


if __name__ == "__main__":
    main()
