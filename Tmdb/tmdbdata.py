# Importing Necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Wrangling
# Changing scientific numbers to numeric numbers
pd.options.display.float_format = '{:.2f}'.format

# Display total data frame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 0)
pd.set_option('display.max_columns', None)

# Loading the data and Number of rows and columns
tmdb = pd.read_csv('tmdb dataset.csv')
print(f"Number of Observations in tmdb dataset: {tmdb.shape}")
print(tmdb)

# datatypes
print(tmdb.info())

# Release date in object instead of datetime.
tmdb['release_date'] = pd.to_datetime(tmdb['release_date'])

# extract day and month from release date
tmdb['release_day'] = pd.to_datetime(tmdb['release_date']).dt.day
tmdb['release_month'] = pd.to_datetime(tmdb['release_date']).dt.month

# We have date,month,year in three different columns. Hence deleting release_date column.
tmdb.drop(labels='release_date', axis=1, inplace=True)

# basic descriptive statistics
print(tmdb.describe())
# Notable findings:
# Popularity ranges from 0 - 33, but has an average of 0.6 (33 could be an outlier?)
# Votes range from 1.5 to 9.2 (probably on a scale of 1-10), with an average of 6
# Budget (usd) ranges from approx. 0 - 425 million (average 17.6 million)
# Revenue (usd) ranges from approx. 0 - 2.8 billion (average 51.4 million)
# Release years range from 1960 - 2015 (average 2001, most were released after 1995)

# Data cleaning
tmdb.drop(['imdb_id', 'homepage', 'tagline', 'overview', 'runtime', 'budget_adj', 'revenue_adj'], axis=1, inplace=True)
print(tmdb.head())

# assess if there are any duplicates.
print(sum(tmdb.duplicated()))

# There is only 1 duplicate, so we'll drop that row and perform 2 checks to ensure the duplicates were removed.
tmdb.drop_duplicates(inplace=True)
print(sum(tmdb.duplicated()))
print(tmdb.shape)

# assess if any rows have missing values.
print(tmdb.isnull().sum())

tmdb.dropna(inplace=True)
print(tmdb.isnull().sum().any())
print(tmdb.info())

# add a profit column so we can create a profitability ratio.
# Profit = revenue (aka income) - budget (aka cost or expense)
tmdb['profit'] = tmdb['revenue'] - tmdb['budget']
tmdb.head()

# make sure no negative numbers for profit
tmdb.loc[tmdb['profit'] < 0, 'profit'] = 0

# Now that we have profit column, we can create a profitability ratio column.
# Profitability ratio = (profit/revenue) x 100 = percentage
# Adjust for non-zero division by adding .0001 to the denominator, revenue.
# I'll convert this column from float to integer so we have non-decimal values ranging from 1-100.
tmdb['profitability_ratio'] = (tmdb['profit'] / (tmdb['revenue'] + .0001)) * 100
tmdb['profitability_ratio'] = tmdb['profitability_ratio'].astype(int)
tmdb.sort_values(['profitability_ratio'], ascending=False).tail()

print(tmdb['profitability_ratio'].nunique())

tmdb.loc[tmdb['profitability_ratio'] < 0, 'profitability_ratio'] = 0
print(tmdb['profitability_ratio'].nunique())

# Revenue column into groups: low (under a million), mediun (millions), and high (billions).
bin_edges = [0, 1e+06, 1e+09, 2.827124e+09]
bin_names = ['under_million', 'millions', 'billions']
tmdb['revenue_rating'] = pd.cut(tmdb['revenue'], bin_edges, labels=bin_names)
tmdb.head()

print(tmdb['revenue_rating'].value_counts())

print(tmdb.isnull().sum())

# To clean up the revenue rating, make all the rows with null values 0 since those rows have no revenue or budget
tmdb.revenue_rating.fillna('under_million', inplace=True)
tmdb.info()

# The release years range from 1960 to 2015. a column for all the decades.
bin_edges = [1959, 1970, 1980, 1990, 2000, 2010, 2015]
bin_names = ['sixties', 'seventies', 'eighties', 'nineties', 'two_thousands', 'two_thousand_tens']
tmdb['decades'] = pd.cut(tmdb['release_year'], bin_edges, labels=bin_names)
print(tmdb.head())

# split

# create separate dataframes for each: genres, cast, and director.
tmdb['genres'].str.contains('|')
tmdb['genres'].nunique()
# Remove the 'genres' column (with multiple values) and replace it with a 'genre' column (with single values).
tmdb_split_genre = tmdb.copy()
split_genre = tmdb_split_genre['genres'].str.split('|').apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
split_genre.name = 'genre_split'
tmdb_split_genre = tmdb_split_genre.drop(['genres'], axis=1).join(split_genre)
print(tmdb_split_genre)

# The genres is now split up and stacked,
# checking to make sure that the new genre column contains only single values.
print(tmdb_split_genre['genre_split'].unique())

# check for duplicates and view the info for the new dataset.
print(tmdb_split_genre.info())
print(tmdb_split_genre.shape)
print(sum(tmdb_split_genre.duplicated()))

# checking for null values
print(tmdb_split_genre.isnull().sum())

# repeat same procedure for cast and director
# split for cast
tmdb_split_cast = tmdb.copy()
split_cast = tmdb_split_cast['cast'].str.split('|').apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
split_cast.name = 'cast_split'
tmdb_split_cast = tmdb_split_cast.drop(['cast'], axis=1).join(split_cast)
print(tmdb_split_cast)
print(tmdb_split_cast.info())
print(tmdb_split_cast.shape)
print(sum(tmdb_split_cast.duplicated()))

# split for director
tmdb_split_director = tmdb.copy()
split_director = tmdb_split_director['director'].str.split('|').apply(pd.Series, 1).stack().reset_index(level=1,
                                                                                                        drop=True)
split_director.name = 'director_split'
tmdb_split_director = tmdb_split_director.drop(['director'], axis=1).join(split_director)
print(tmdb_split_director)
print(tmdb_split_director.info())
print(tmdb_split_director.shape)
print(sum(tmdb_split_director.duplicated()))

print(tmdb.info())


# Exploratory Analysis

# What genres are most popular overall?
tmdb_split_genre['genre_split'].value_counts().plot(kind='bar', color='blue')
plt.title('Movies by Genre, 1960-2015', size=18)
plt.xlabel('Genre', size=12)
plt.ylabel('Movie count', size=10)
plt.show()

# view with a pie chart
tmdb_split_genre['genre_split'].value_counts().plot(kind='pie', figsize=(7, 10))
plt.show()

# What genres are most popular throughout the decades?
genres_decades = tmdb_split_genre.groupby(['decades'])['genre_split'].value_counts()
genres_decades_largest = genres_decades.groupby(level=0).nlargest(3).reset_index(level=0, drop=True)
print(genres_decades_largest)

# What properties are associated with higher revenues?
# General scatter plots of revenue vs budget, profit, profitability_ratio and popularity.
# Revenue vs Budget
tmdb.plot(x='revenue', y='budget', kind='scatter')
plt.show()
# Revenue vs Profit
tmdb.plot(x='revenue', y='profit', kind='scatter', color='brown')
plt.show()
# Revenue vs profitability ratio
tmdb.plot(x='revenue', y='profitability_ratio', kind='scatter', color='grey')
plt.show()
# Revenue vs popularity
tmdb.plot(x='revenue', y='popularity', kind='scatter', color='green')
plt.show()

# Revenue and budget have a weak positive correlation.
# Revenue and profit have a strong positive correlation.
# Revenue and profitability ratio have a weak positive correlation.
# Revenue and popularity have positive correlation, movies with higher revenues tend tobe more popular.


# Which actors have starred in the most movies?
cast = tmdb_split_cast['cast_split'].value_counts().head(20)
print(cast)

# Who has directed the most movies?
director = tmdb_split_director['director_split'].value_counts().head(20)
print(director)

# What are the most popular movies?
popular_movies = tmdb[['popularity', 'original_title']].sort_values(by='popularity', ascending=False).head(10)
print(popular_movies)
