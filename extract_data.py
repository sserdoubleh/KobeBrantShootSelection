import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "data.csv"
df = pd.read_csv(filename)

# initialize no_needed
no_needed = []

# functions
def calculate_accuracy(raw):
    return 1.0 * raw['shot_id'][raw.shot_made_flag==1].count() / raw['shot_id'][raw.shot_made_flag.notnull()].count()

# time remaining
df['time_remaining'] = df['minutes_remaining'] * 60 + df['seconds_remaining']
df['key_shot'] = np.logical_and(df['time_remaining'] <= 3, df['period'] >= 4).astype(int)
df['time_remaining'] = np.sqrt(df['time_remaining']).astype(int)
no_needed.extend(['minutes_remaining', 'seconds_remaining'])

# shot distance and location
df['shot_distance'] = np.sqrt((df['loc_x'] / 10) ** 2 + (df['loc_y'] / 10) ** 2).astype(int)
loc_x_zero = df['loc_x'] == 0

# delete angle~~~
# df['shot_angle'] = np.array([0] * len(df))
# df['shot_angle'][~loc_x_zero] = np.arctan(df['loc_y'][~loc_x_zero] / df['loc_x'][~loc_x_zero])
# df['shot_angle'][loc_x_zero] = np.pi / 2
no_needed.extend(['lat', 'lon', 'loc_x', 'loc_y'])

# shot zone
shot_zone_types = ['shot_zone_area', 'shot_zone_basic', 'shot_zone_range']
for shot_zone_type in shot_zone_types:
    for shot_type in df[shot_zone_type].unique():
        df[shot_type] = df[shot_zone_type].str.contains(shot_type).astype(int)
no_needed.extend(shot_zone_types)

# shot type
attributes = ['action_type', 'combined_shot_type']
for attribute in attributes:
    for shot_type in df[attribute].unique():
        df[shot_type] = df[attribute].str.contains(shot_type).astype(int)
df['3PT'] = df['shot_type'].str.contains('3PT').astype(int)
attributes.append('shot_type')
no_needed.extend(attributes)

# season
df['season'] = df['season'].apply(lambda x: int(x.split('-')[1]))

# opponent
for opponent in df['opponent'].unique():
	df[opponent] = (df.opponent==opponent).astype(int)
no_needed.append('opponent')

df['court'] = np.array([0] * len(df))
home = df['matchup'].str.contains('vs.')
df['court'][home] = "LAL"
df['court'][~home] = df['matchup'][~home].apply(lambda x: x.split(' @ ')[1])
for court in df['court'].unique():
	df['court ' + str(court)] = (df.court==court).astype(int)
no_needed.append('matchup')
no_needed.append('court')

# no need
df.set_index('shot_id', inplace=True)
no_needed.extend(['team_id', 'team_name', 'game_event_id', 'game_id'])

# game date & rest time
df['game_date'] = pd.to_datetime(df['game_date'])
df['game_date'] = (df['game_date'] - df['game_date'].min()).astype('timedelta64[D]').astype(int) + 1

# game_dates = []
# last_game_dates = []
# 
# from datetime import timedelta
# for season in df['season'].unique():
#     game_date = df['game_date'][df.season==season].unique()
#     game_dates.extend(game_date)
#     first_date = np.datetime64(pd.to_datetime(game_date[0]) - timedelta(days=30))
#     last_game_date = np.concatenate((np.array([first_date]), game_date[:game_date.size - 1]))
#     last_game_dates.extend(last_game_date)
# new_df = pd.DataFrame({'game_date': game_dates, 'last_game_date': last_game_dates})
# df = pd.merge(df, new_df, on='game_date')
# df['rest_time'] = df['game_date'] - df['last_game_date']
# df['rest_time'] = df['rest_time'].dt.days
# no_needed.append('last_game_date')

no_needed.append('game_date')

df = df.drop(no_needed, axis=1)
print df.dtypes
df.to_csv('in', index=False)
