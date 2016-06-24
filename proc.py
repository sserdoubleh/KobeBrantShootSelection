import numpy as np
import pandas as pd

# initialize
df = pd.read_csv("data.csv")
no_needed = []

# functions
def calculate_accuracy(labels):
    has = df[df['shot_made_flag'].notnull()]
    result = has.groupby(labels)['shot_made_flag']
    print result.value_counts()
    print result.mean()

# date (new added)
df['game_date'] = pd.to_datetime(df['game_date'])

# time remaining
df['time_remaining'] = df['minutes_remaining'] * 60 + df['seconds_remaining']
df['key_shot'] = (df['time_remaining'] < 10).astype(int)
no_needed.extend(['seconds_remaining', 'time_remaining'])

# shot distance
df['shot_distance'] = df['shot_distance'].astype(int)
no_needed.extend(['lat', 'lon', 'loc_x', 'loc_y'])

# court (new added)
df['court'] = df['matchup'].apply(lambda x: 'LAL' if 'vs.' in x else x.split(' @ ')[1])
no_needed.append('matchup')

# df['shot_zone_area2'] = df['shot_zone_area'].str.contains('Center').astype(int)

action_set = set()

# split attributes
attributes = ['action_type', 'combined_shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'opponent', 'court']
for attribute in attributes:
    df = pd.concat([df, pd.get_dummies(df[attribute], prefix=attribute)], axis=1)
no_needed.extend(attributes)

# key word list (new added)
action_attributes = ['action_type', 'combined_shot_type']
for attribute in action_attributes:
    for value in df[attribute].unique():
        for key_word in value.split(' '):
            action_set.add(key_word)
for key_word in action_set:
    df['action_' + key_word] = np.logical_or(df['action_type'].str.contains(key_word), df['combined_shot_type'].str.contains(key_word)).astype(int)

# 3PT
df['3PT'] = df['shot_type'].str.contains('3PT').astype(int)
no_needed.append('shot_type');

# season
df['season'] = df['season'].apply(lambda x: int(x.split('-')[0]))
df['season'] = (df['season'] - df['season'].min()).astype(int)

calculate_accuracy('season')

# no need
df.set_index('shot_id', inplace=True)
no_needed.extend(['team_id', 'team_name', 'game_event_id', 'game_id', 'game_date'])

df = df.drop(no_needed, axis=1)
# df.to_csv('my.in', index=False)
