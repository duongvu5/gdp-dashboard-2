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

def filter_duplicate_players_gk(df):
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

def get_stats_lists_gk():
    """Return lists of stats categories."""
    stats_list = ['standard', 'shooting', 'passing', 'passing_types', 'goal_shot_creation', 'defense', 'possession', 'misc', 'keeper', 'keeper_adv']
    keeper_stats_list = []
    non_related_list = ['playing_time']
    return stats_list, keeper_stats_list, non_related_list

def read_and_filter_stats_gk(fbref, stats_list):
    stats_list = ['standard',
         'shooting',
             'passing',
             'passing_types',
            'goal_shot_creation',
             'defense',
             'possession',
             'misc',
              'keeper',
            'keeper_adv'
             ]
    df_list1 = []
    df_list2 = []
    df1 = fbref.read_player_season_stats('standard')
    gk_list = df1[df1['pos'] == 'GK']['player'].to_list()
    df = df1[df1['player'].isin(gk_list)]
    df_normal, group_combined = filter_duplicate_players_gk(df)
    df_list1.append(df_normal)
    df_list2.append(group_combined)
    dropped_columns = ['nation', 'pos', 'team', 'age', 'born' , 'league', 'season']
    for i in range(1, len(stats_list)):
        df = fbref.read_player_season_stats(stats_list[i])
        df.drop(columns = dropped_columns, inplace = True)
        df.fillna(0, inplace = True)
        df_filtered = df[df['player'].isin(gk_list)]
        df_normal, df_combined = filter_duplicate_players_gk(df)
        df_list1.append(df_normal)
        df_list2.append(df_combined)
    
    return df_list1, df_list2