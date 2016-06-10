import numpy as np
import pandas as pd

# initialize
df = pd.read_csv("data.csv")
no_needed = []

# functions
def calculate_accuracy(label):
    has = df[df['shot_made_flag'].notnull()]
    result = has.groupby(label)['shot_made_flag']
    print result.value_counts()
    print result.mean()

# time remaining
df['time_remaining'] = df['minutes_remaining'] * 60 + df['seconds_remaining']
df['key_shot'] = (df['time_remaining'] < 10).astype(int)
no_needed.extend(['seconds_remaining', 'time_remaining'])

# shot distance
df['shot_distance'] = df['shot_distance'].astype(int)
no_needed.extend(['lat', 'lon', 'loc_x', 'loc_y'])

# split attributes
attributes = ['action_type', 'combined_shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'shot_type', 'opponent']
for attribute in attributes:
    df = pd.concat([df, pd.get_dummies(df[attribute], prefix=attribute)], axis=1)
no_needed.extend(attributes)

# season
df['season'] = df['season'].apply(lambda x: int(x.split('-')[0]))

df['home'] = df['matchup'].str.contains('vs.').astype(int)
no_needed.append('matchup')

# no need
df.set_index('shot_id', inplace=True)
no_needed.extend(['team_id', 'team_name', 'game_event_id', 'game_id', 'game_date'])

df = df.drop(no_needed, axis=1)
df.to_csv('my.in', index=False)
