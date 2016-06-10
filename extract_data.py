import numpy as np
import pandas as pd

# initialize
df = pd.read_csv("data.csv")

# functions
def calculate_accuracy(label):
    has = df[df['shot_made_flag'].notnull()]
    result = has.groupby(label)['shot_made_flag']
    print result.value_counts()
    print result.mean().sort_values()

df['away'] = df['matchup'].str.contains('@')

# time remaining
df['time_remaining'] = df['minutes_remaining'] * 60 + df['seconds_remaining']
df['time_remaining'] = np.sqrt(df['time_remaining']).astype(int)
df['key_shot'] = df['time_remaining'] < 3

actiontypes = df.action_type.value_counts()
df['type'] = df.apply(lambda row: row['action_type'] if actiontypes[row['action_type']] > 20 else row['combined_shot_type'], axis=1)

# shot distance
df['shot_distance'] = df['shot_distance'].apply(lambda x: int(x if x < 45 else 45))

# season
df['season'] = df['season'].apply(lambda x: int(x.split('-')[0]))

df['home'] = df['matchup'].str.contains('vs.').astype(int)

# no need
df.set_index('shot_id', inplace=True)

features = ['away', 'period', 'playoffs', 'season',\
        'type', 'shot_type', 'opponent',\
        'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',\
        'minutes_remaining', 'shot_distance', 'key_shot']

data = df.loc(:, ['shot_id', 'shot_made_flag'])
for f in features:
    data = pd.concat([data, pd.get_dummies(df[f], prefix=f)], axis=1)

data.to_csv('my.in', index=False)
