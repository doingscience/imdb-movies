import csv
import numpy as np

import pandas as pd

column_types = {
    'isAdult': float,
    'startYear': float,
    'endYear': float,
    'runtimeMinutes': float,
    'tconst': str,
    'titleType': str,
    'primaryTitle': str,
    'originalTitle': str,
    'genres': str
}

titles_df = pd.read_csv("https://datasets.imdbws.com/title.basics.tsv.gz",
                        dtype=column_types,
                        na_values=r'\N',
                        sep="\t",
                        quoting=csv.QUOTE_NONE)

for col in titles_df:
    print(titles_df[col].unique())

print(titles_df.shape)

titles_df.head()
titles_df.iloc[0]

titles_df.isna().sum()

titles_df['titleType'].unique()

tv_types = [
    'tvMovie',
    'tvSeries',
    'tvEpisode',
    'tvShort',
    'tvMiniSeries',
    'tvSpecial'
]
titles_df = titles_df.loc[titles_df['titleType'].isin(tv_types)]
titles_df = titles_df.loc[titles_df['primaryTitle'] == titles_df['originalTitle']]
titles_df.loc[titles_df['startYear'].idxmin()]

titles_df = titles_df[titles_df['startYear'] >= 1945]

column_types = {
    'seasonNumber': float,
    'episodeNumber': float,
    'tconst': str,
    'parentTconst': str
}

episodes_df = pd.read_csv("https://datasets.imdbws.com/title.episode.tsv.gz",
                          dtype=column_types,
                          na_values=r'\N',
                          sep="\t",
                          quoting=csv.QUOTE_NONE)

print(episodes_df.shape)
episodes_df.head()

titles_episodes_df = pd.merge(titles_df, episodes_df,
                              left_on='tconst',
                              right_on='parentTconst',
                              suffixes=['_ti', '_ep'])
titles_episodes_df.shape

titles_episodes_df.head()
titles_episodes_df = titles_episodes_df.drop('tconst_ti', axis=1)
titles_episodes_df.rename(columns={'tconst_ep': 'tconst'}, inplace=True)
titles_episodes_df.head()

titles_episodes_df.iloc[0]

titles_episodes_df[titles_episodes_df['primaryTitle'] == 'Better Call Saul'].iloc[0]
titles_episodes_df = titles_episodes_df.drop(['titleType',
                                              'originalTitle',
                                              'isAdult',
                                              'runtimeMinutes',
                                              'genres'], axis=1)
titles_episodes_df.head()

import datetime

cur_year = int(datetime.datetime.now().year)

titles_episodes_df['endYear'].fillna(cur_year, inplace=True)

titles_episodes_df['seasonNumber'].fillna(1, inplace=True)

titles_episodes_df.head()

ratings_df = pd.read_csv("https://datasets.imdbws.com/title.ratings.tsv.gz",
                         dtype=column_types,
                         na_values=r'\N',
                         sep="\t")
ratings_df.iloc[0]
print(ratings_df.shape)
ratings_df.head()

titles_episodes_rankings_df = pd.merge(titles_episodes_df,
                                       ratings_df,
                                       on='tconst')
titles_episodes_rankings_df.head()
titles_episodes_rankings_df.iloc[0]
seasons = titles_episodes_df[['parentTconst', 'seasonNumber']] \
    .groupby(['parentTconst']).max()
seasons.reset_index(inplace=True)
seasons.head()

seasons.rename(columns={'seasonNumber': 'numSeasons'}, inplace=True)
seasons.head()

titles_episodes_rankings_df = pd.merge(titles_episodes_rankings_df, seasons, on='parentTconst')
titles_episodes_rankings_df.head()


titles_episodes_rankings_df['year'] = titles_episodes_rankings_df.apply(
    lambda x: np.linspace(int(x['startYear']),
                          int(x['endYear']),
                          int(x['numSeasons']))[int(x['seasonNumber']) - 1],
    axis=1
)


titles_episodes_rankings_df.query('primaryTitle == "Game of Thrones"')[::8]

to_show = titles_episodes_rankings_df[
    ['parentTconst',
     'primaryTitle',
     'seasonNumber',
     'year',
     'startYear',
     'numSeasons',
     'averageRating',
     'numVotes']] \
    .groupby(['parentTconst', 'primaryTitle', 'seasonNumber', 'year', 'numSeasons', 'startYear']) \
    .agg({'averageRating': 'mean',
          'numVotes': 'sum'}).reset_index()
print(to_show.shape)
to_show.head()

to_show = to_show.query('(averageRating >= 5) & (startYear >= 1990) & (year <= 2021)')
print(to_show.shape)
to_show.head()

avg_votes_gt_1000 = to_show.groupby('parentTconst').agg({'numVotes': 'mean'}).query('numVotes >= 1000')
print(avg_votes_gt_1000.shape)
avg_votes_gt_1000.head()

to_show = pd.merge(to_show, avg_votes_gt_1000, how='right', left_on='parentTconst', right_index=True)
to_show.rename(columns={'numVotes_x': 'numVotes', 'numVotes_y': 'avgVotes'}, inplace=True)
print(to_show.shape)
to_show.head()

to_show['intYear'] = to_show['year'].astype(int)
to_show.head()

votes_per_year = to_show[['intYear', 'numVotes']].groupby('intYear').sum()
votes_per_year.rename(columns={'numVotes': 'yearVotes'}, inplace=True)
votes_per_year.head()

votes_per_year.tail()

to_show = pd.merge(to_show, votes_per_year, left_on='intYear', right_index=True)
to_show['propVotes'] = 100 * to_show['numVotes'] / to_show['yearVotes']
to_show.sample(n=10, random_state=42)

to_show.to_csv('to_show.csv')

lines = to_show[
    [
        'parentTconst', 'seasonNumber', 'year', 'averageRating'
    ]].groupby(['parentTconst']) \
    .agg({'year': list, 'averageRating': list})


def sort_years_ratings(row):
    arr_years = np.array(row['year'])
    sorted_indices = np.argsort(arr_years)
    arr_years = arr_years[sorted_indices]
    arr_ratings = np.array(row['averageRating'])
    arr_ratings = arr_ratings[sorted_indices]
    row['year'] = arr_years
    row['averageRating'] = arr_ratings
    return row


lines.apply(sort_years_ratings, axis=1)
lines.to_json('lines.json', orient='index')
