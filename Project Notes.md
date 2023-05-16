**Dataset**

I scraped my dataset from VGChartz.com. The dataset is the most extensive one that I could find online and as of 2018, the no longer produce estimates for sales but rather record sale data where the data is made available by developers and publishers. I used another github project as a basis for how I scraped the data however I modified it to include genre – this was done by creating a file for each genre then amalgamating all of them. I also got a Steam dataset from Kaggle to confirm which games were on Steam and which were not for PC/Mac.

These datasets were chosen due to their size and accuracy.

VGChartz dataset description:

This dataset includes 62735 records and 17 columns:

 - Source.Name: The CSV that the data came from which notes which genre the game is in
 - Position: The rank within the genre
 - game: The game name
 - console: The console the game was released on (note that games may have been released on multiple consoles)
 - publisher: The publisher of the game
 - developer: The developer of the game
 - vgchart_score: A score assigned by VG Chartz, the provider of the dataset
 - critic_score: An amalgamation of scores provided by critics done by VG Chartz
 - user_score: An amalgamation of scores provided by users done by VG Chartz
 - total_shipped: The total number of copies of a game shipped
 - total_sales: The total number of sales in millions over all regions
 - na_sales: The number of sales in millions in the North America region
 - pal_sales: The number of sales in millions in the PAL region which covers many European countries, Australia and New Zealand
 - japan_sales: The number of sales in millions in the Japan region
 - other_sales: The The number of sales in millions any other region not covered by NA/PAL/Japan
 - release_date: The date the game was released
 - last_update: The date the figures were last updated

Steam dataset description:

This dataset includes 27076 records and 18 columns:

 - appid: Unique identifier within the dataset
 - name: The game name
 - release_date: The date the game was released
 - english: If the game is in English or not
 - developer: The developer of the game
 - publisher: The publisher of the game
 - platforms: The platforms the game is available in (windows/mac/linux)
 - required_age: The minimum age you have to be to play the game
 - categories: The playable category of the game (single player/multi player/virtual support etc)
 - genres: The genre of the game
 - steamspy_tags: A list of searchable tags on the game
 - achievements: How many achievements a player can complete
 - positive_ratings: How many positive ratings the game has
 - negative_ratings: How many negative ratings the game has
 - average_playtime: The average playtime over all players that have played the game
 - median_playtime: The median playtime over all players that have played the game
 - owners: How many players own the game
 - price: The price of the game

**Import Library**
<code> 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier
</code>

**Import Files and overview**
<code> 
#import the csv files
df_original = pd.read_csv(r'C:\Users\heath\Final Webscrape\RawData.csv')
df_original_steam = pd.read_csv(r'C:\Users\heath\Final Webscrape\steam.csv')

#Overview of the data before anything has been done to it
print('Original dataset information:')
print(df_original.shape)
print(df_original.head(5))
print(df_original.info())
print(df_original.describe())

print('Original steam dataset information:')
print(df_original_steam.shape)
print(df_original_steam.head(5))
print(df_original_steam.info())
print(df_original_steam.describe()) 
</code>
