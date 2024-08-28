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
    df_list = []
    df1 = fbref.read_player_season_stats('standard')
    gk_list = df1[df1['pos'] == 'GK']['player'].to_list()
    df1_filtered = df1[df1['player'].isin(gk_list)]
    df_list.append(df1_filtered)
    dropped_columns = ['nation', 'pos', 'team', 'age', 'born' , 'league', 'season']
    for i in range(1, len(stats_list)):
        df = fbref.read_player_season_stats(stats_list[i])
        df.drop(columns = dropped_columns, inplace = True)
        df.fillna(0, inplace = True)
        df_filtered = df[df['player'].isin(gk_list)]
        df_list.append(df_filtered)
    
    return df_list
