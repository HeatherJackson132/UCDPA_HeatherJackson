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
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier
```

**Import Files and overview**
```
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
```
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/2d9076b1-9ea4-4ca4-bc50-ece6f107977f)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/11b3197b-7b50-439f-9909-e31120b97464)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/b6d63043-39a8-4edb-9c07-9dd237245bf5)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/5a674e62-6c20-4e48-9958-8bcd32b87a69)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/870f77a8-ecc6-49f0-8647-455ee5f3bfc8)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/369b6f38-e87c-417c-a8ba-aa337c1990a8)

**Data Clean Up**
```

#remove columns that I don't require for this project. I removed position(as it wasn't a unique id), the VG Chart score, total shipped and last updated. More may need to be removed and/or put back
df = df_original.drop(columns=['position','vgchart_score','total_shipped','last_update']) 

#remove any rows where there are no sales as I could see from df_original.info that there were only 18977 populated fields 
df = df.dropna(subset=['total_sales'])

df = df.replace(r"^ +| +$", r"", regex=True) #remove any spaces at start or end of fields
df.game = df.game.str.replace('[^a-zA-Z0-9 ]', '', regex=True) # remove any non-alphanumeric characters

no_date_count = df['release_date'].value_counts()['N/A']
print('The number of rows with no release date:', no_date_count)

#remove any rows where there's no release date as there are only 80
df = df[df['release_date'].str.contains('N/A')==False]


#rename source.name with 'genre' and remove my file names so only the genre information is left
df.rename(columns={'Source.Name':'genre'},inplace=True)
df['genre']=df['genre'].str.replace(r'HeathersVGchartzDatabase','', regex=True) #removes HeathersVGchartzDatabase from start
df['genre']=df['genre'].str.replace(r'.csv','', regex=True) #removes .csv from end



#update all text fields to string
df['genre'] = df['genre'].astype('string')
df['game'] = df['game'].astype('string')
df['console'] = df['console'].astype('string')
df['publisher'] = df['publisher'].astype('string')
df['developer'] = df['developer'].astype('string')


#remove m from all the sales columns so that they are numeric values only and update the column names to say (million)
df.rename(columns={'total_sales':'total_sales_millions'},inplace=True)
df['total_sales_millions']=df['total_sales_millions'].str.replace(r'm','', regex=True)

df.rename(columns={'na_sales':'na_sales_millions'},inplace=True)
df['na_sales_millions']=df['na_sales_millions'].str.replace(r'm','', regex=True)

df.rename(columns={'pal_sales':'pal_sales_millions'},inplace=True)
df['pal_sales_millions']=df['pal_sales_millions'].str.replace(r'm','', regex=True)

df.rename(columns={'japan_sales':'japan_sales_millions'},inplace=True)
df['japan_sales_millions']=df['japan_sales_millions'].str.replace(r'm','', regex=True)

df.rename(columns={'other_sales':'other_sales_millions'},inplace=True)
df['other_sales_millions']=df['other_sales_millions'].str.replace(r'm','', regex=True)

#0 implies no sales however in this context, it actually means less than 0.005 million (due to rounding, anything over that would have rounded to 0.01 as the data only goes to 2 decimal places)
#to make the graphs look more sensible (ie not show 0 sales when there should be at least some) I will replace any sales that are 0 with 0.0025 - half way between 0 and 0.005)
df['total_sales_millions']=df['total_sales_millions'].str.replace('0.00','0.0025', regex=True)
df['na_sales_millions']=df['na_sales_millions'].str.replace('0.00','0.0025', regex=True)
df['pal_sales_millions']=df['pal_sales_millions'].str.replace('0.00','0.0025', regex=True)
df['japan_sales_millions']=df['japan_sales_millions'].str.replace('0.00','0.0025', regex=True)
df['other_sales_millions']=df['other_sales_millions'].str.replace('0.00','0.0025', regex=True)

#convert all sales fields to floats
df['total_sales_millions'] = df['total_sales_millions'].astype('float')
df['na_sales_millions'] = df['na_sales_millions'].astype('float')
df['pal_sales_millions'] = df['pal_sales_millions'].astype('float')
df['japan_sales_millions'] = df['japan_sales_millions'].astype('float')
df['other_sales_millions'] = df['other_sales_millions'].astype('float')


#create fields for the month and year the games were released using regex to split them
#all of the elements of the date have the same position so can just isolate the month characters already there
df[['drop0','release_month','drop1','temp_year']]=df['release_date'].str.extract(r"(\d{2}\w{2}\s)(\w{3})(\s)(\d{2})",expand=True)
df=df.drop(columns=['drop0','drop1'])



#take any leading zeros away from the years
df['temp_year']=df['temp_year'].str.replace(r'00','0', regex=True)
df['temp_year']=df['temp_year'].str.replace(r'01','1', regex=True)
df['temp_year']=df['temp_year'].str.replace(r'02','2', regex=True)
df['temp_year']=df['temp_year'].str.replace(r'03','3', regex=True)
df['temp_year']=df['temp_year'].str.replace(r'04','4', regex=True)
df['temp_year']=df['temp_year'].str.replace(r'05','5', regex=True)
df['temp_year']=df['temp_year'].str.replace(r'06','6', regex=True)
df['temp_year']=df['temp_year'].str.replace(r'07','7', regex=True)
df['temp_year']=df['temp_year'].str.replace(r'08','8', regex=True)
df['temp_year']=df['temp_year'].str.replace(r'09','9', regex=True)

#change the month to a string and the year to an int
df['release_month'] = df['release_month'].astype('string')
df['temp_year'] = df['temp_year'].astype('int')

#change the year to the 4 character naming convention
newyear=[]
for create_year in df['temp_year']:
    if create_year>= 70:
        newyear.append(create_year + 1900)
    else:
        newyear.append(create_year + 2000)
     

df['release_year']=newyear        
df['release_year'] = df['release_year'].astype('int')
df=df.drop(columns=['temp_year'])


#add a column that groups the various consoles with their manufacturer using a dictionary mapping
df['console_manufacturer'] = df['console'].map({'OSX':'Computer','PC':'Computer','X360':'Microsoft','XB':'Microsoft','XBL':'Microsoft','XOne':'Microsoft','3DS':'Nintendo','DS':'Nintendo','GB':'Nintendo','GBA':'Nintendo','GBC':'Nintendo','GC':'Nintendo','N64':'Nintendo','NES':'Nintendo','NS':'Nintendo','SNES':'Nintendo','VC':'Nintendo','Wii':'Nintendo','WiiU':'Nintendo','WW':'Nintendo','2600':'Other','3DO':'Other','Mob':'Other','NG':'Other','PCE':'Other','PCFX':'Other','WS':'Other','DC':'Sega','GEN':'Sega','GG':'Sega','SAT':'Sega','SCD':'Sega','PS':'Sony','PS2':'Sony','PS3':'Sony','PS4':'Sony','PSN':'Sony','PSP':'Sony','PSV':'Sony'})
df['console_manufacturer'] = df['console_manufacturer'].astype('string')

#add a column that groups the sales values
def salesgroup(temp_sales):
    if temp_sales < 0.05:
        return 'sales < 0.05'
    elif  0.05 <= temp_sales < 0.1:
        return '0.05 < sales 0.1'
    elif 0.1 <= temp_sales < 0.15:
        return '0.1 < sales 0.15'
    elif 0.15 <= temp_sales < 0.2:
        return '0.15 < sales 0.2'    
    elif 0.2 <= temp_sales < 0.25:
        return '0.2 < sales 0.25'    
    elif 0.25 <= temp_sales < 0.3:
        return '0.25 < sales 0.3'    
    elif 0.3 <= temp_sales < 0.35:
        return '0.3 < sales 0.35'    
    elif 0.35 <= temp_sales < 0.4:
        return '0.35 < sales 0.4'    
    elif 0.4 <= temp_sales < 0.45:
        return '0.4 < sales 0.45'    
    elif 0.45 <= temp_sales < 0.5:
        return '0.45 < sales 0.5'    
    elif 0.5 <= temp_sales < 0.55:
        return '0.5 < sales 0.55'    
    elif 0.55 <= temp_sales < 0.6:
        return '0.55 < sales 0.6'    
    elif 0.6 <= temp_sales < 0.65:
        return '0.6 < sales 0.65'    
    elif 0.65 <= temp_sales < 0.7:
        return '0.65 < sales 0.7'   
    elif 0.7 <= temp_sales < 0.75:
        return '0.7 < sales 0.75'   
    elif 0.75 <= temp_sales < 0.8:
        return '0.75 < sales 0.8'   
    elif 0.8 <= temp_sales < 0.85:
        return '0.8 < sales 0.85'
    elif 0.85 <= temp_sales < 0.9:
        return '0.85 < sales 0.9'   
    elif 0.9 <= temp_sales < 0.95:
        return '0.9 < sales 0.95'   
    elif 0.95 <= temp_sales < 1:
        return '0.95 < sales 1'   
    elif 1 <= temp_sales < 2:
        return '1 < sales 2'   
    elif 2 <= temp_sales < 3:
        return '2 < sales 3'   
    elif 3 <= temp_sales < 4:
        return '3 < sales 4'   
    elif 4 <= temp_sales < 5:
        return '4 < sales 5'   
    elif 5 <= temp_sales < 6:
        return '5 < sales 6'   
    elif 6 <= temp_sales < 7:
        return '6 < sales 7'   
    elif 7 <= temp_sales < 8:
        return '7 < sales 8'     
    elif 8 <= temp_sales < 9:
        return '8 < sales 9'   
    elif 9 <= temp_sales < 10:
        return '9 < sales 10'   
    elif 10 <= temp_sales < 11:
        return '10 < sales 11'   
    elif 11 <= temp_sales < 12:
        return '11 < sales 12'   
    elif 12 <= temp_sales < 13:
        return '12 < sales 13'   
    elif 13 <= temp_sales < 14:
        return '13 < sales 14'    
    elif 14 <= temp_sales < 15:
        return '14 < sales 15'    
    elif 15 <= temp_sales < 20:
        return '15 < sales 20'   
    elif 20 <= temp_sales:
        return 'sales > 20'    
 
df['salesgroup'] = df['total_sales_millions'].map(salesgroup) 

print('The counts for each total sales grouping')
print(df['salesgroup'].value_counts().sort_index())

#This was broken up too much. Instead I changed it to just 5 groups. I also numbered the groups so that they would display in a sensible order. I kept this in as I thought it might be useful later 

def salesgrouping(temp_sales2):
    if temp_sales2 < 0.05:
        return '1. Very small'
    elif  0.05 <= temp_sales2 < 0.2:
        return '2. Small'
    elif  0.2 <= temp_sales2 < 1:
        return '3. Medium'
    elif  1 <= temp_sales2 < 10:
        return '4. Large'
    elif 10 <= temp_sales2:
        return '5. Very Large'  

df['salesgrouping'] = df['total_sales_millions'].map(salesgrouping) 

#adding steam data to pc data only
#there are duplicate rows in the steam dataset (which I was going to eliminate by doing a concat with the developer publisher but none of the duplicates from the steam dataset are in the main dataset)
#only keeping the app id - unique id, platforms and ratings
df_steam = df[df['console_manufacturer'].str.contains('Computer')==True]


#update the steam data so that there are no special characters or unexpected spaces
df_clean_steam = df_original_steam
df_clean_steam = df_clean_steam.replace(r"^ +| +$", r"", regex=True) #remove any spaces at start or end of fields
df_clean_steam.name = df_clean_steam.name.str.replace('[^a-zA-Z0-9 ]', '', regex=True) # remove any non-alphanumeric characters



df_steam=pd.merge(left=df_steam,right=df_clean_steam,left_on='game', right_on='name',how='left',indicator='steam')
#drop steam data that's not needed
df_steam = df_steam.drop(columns=['name','release_date_y','english','developer_y','publisher_y','required_age','categories','genres','steamspy_tags','achievements','average_playtime','median_playtime','owners','price']) 
#update steam columns names
df_steam.rename(columns={'appid':'steam_id'},inplace=True)
df_steam.rename(columns={'platforms':'steam_platforms'},inplace=True)
df_steam.rename(columns={'positive_ratings':'steam_positive_ratingss'},inplace=True)
df_steam.rename(columns={'negative_ratings':'steam_negative_ratings'},inplace=True)


df_steam['steam']=df_steam['steam'].str.replace(r'left_only','No', regex=True)
df_steam['steam']=df_steam['steam'].str.replace(r'both','Yes', regex=True)


df_steam['steam_platforms'] = df_steam['steam_platforms'].astype('string')
df_steam['steam'] = df_steam['steam'].astype('string')


#create a subset for just the critic/user scores - a much smaller group so don't want to reduce the full dataset when it isn't going to be used much
#removed all rows with N/A for the critic/user scores
df_scores = df[df['critic_score'].str.contains('N/A')==False]
df_scores = df_scores[df_scores['user_score'].str.contains('N/A')==False]

#update the scores to floats
df_scores['critic_score'] = df_scores['critic_score'].astype('float')
df_scores['user_score'] = df_scores['user_score'].astype('float')

#Overview of the data following amendments

print('New dataset information:')
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

print('New steam dataset information:')
print(df_steam.shape)
print(df_steam.head())
print(df_steam.info())
print(df_steam.describe())

print('New scores dataset information:')
print(df_scores.shape)
print(df_scores.head())
print(df_scores.info())
print(df_scores.describe())

#I downloaded the files as csvs so that I could easily review them and spot any obvious issues
df.to_csv('currentdata.csv ', index=False, encoding='utf-8')
df_steam.to_csv('currentdata_steam.csv ', index=False, encoding='utf-8')
df_scores.to_csv('currentdata_scores.csv ', index=False, encoding='utf-8')

```
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/90f91798-28c4-4c66-9ef6-aa979ccc9434)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/93f37837-b608-4d96-84e5-87e59a57b133)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/1f8ea2e9-a147-472d-8a8e-7a4ab13124f6)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/534fef72-934a-41a0-ae63-73490aa6a1b1)

**Various Counts or Metrics**
```

#various counts:
    
#main dataset

#about genre:
print('GENRE INFO:')
genre_counts = df['genre'].value_counts()
print('The counts of each genre in the main data set:\n',genre_counts)
genre_means = df.groupby('genre')['total_sales_millions'].mean()
print('The mean sales for every genre:\n',genre_means)
genre_std = df.groupby(['genre'])['total_sales_millions'].std()
print('The std of the sales for every genre:\n',genre_std)


#about manufacturer

print('MANUFACTURER INFO:')
manufacturer_counts = df['console_manufacturer'].value_counts()
print('The counts for each console manufacturer in the main data set:\n',manufacturer_counts)
manufacturer_means = df.groupby('console_manufacturer')['total_sales_millions'].mean()
print('The mean sales for every manutfacturer:\n',manufacturer_means)
manufacturer_std = df.groupby(['console_manufacturer'])['total_sales_millions'].std()
print('The std of the sales for every manutfacturer:\n',manufacturer_std)

#annual information:

print('ANNUAL INFO:') 
annual_counts = df['release_year'].value_counts().sort_index()
print('The counts for each year in the main data set:\n',annual_counts)
annual_means = df.groupby('release_year')['total_sales_millions'].mean()
print('The mean sales for every year:\n',annual_means)
annual_std = df.groupby(['release_year'])['total_sales_millions'].std()
print('The std of the annual sales:\n',annual_std)
#note there is a NaN value here as there was only one game in this year in the dataset

#Sales grouping info:

print('SALES GROUPING INFO:')
salesgrouping_counts = df['salesgrouping'].value_counts().sort_index()
print('The counts for each total sales grouping:\n',salesgrouping_counts)


#monthly information:
print('MONTHLY INFO:')
monthly_counts = df['release_month'].value_counts()
print('The counts for each month:\n',monthly_counts)


#Regional Sales info:

print('REGIONAL INFO:')
na_sales_total = df['na_sales_millions'].sum()
print('The total sales for NA:\n',na_sales_total)
na_sales_mean = df['na_sales_millions'].mean()
print('The average sales for NA:\n',na_sales_mean)

pal_sales_total = df['pal_sales_millions'].sum()
print('The total sales for PAL:\n',pal_sales_total)
pal_sales_mean = df['pal_sales_millions'].mean()
print('The average sales for PAL:\n',pal_sales_mean)

japan_sales_total = df['japan_sales_millions'].sum()
print('The total sales for Japan:\n',japan_sales_total)
japan_sales_mean = df['japan_sales_millions'].mean()
print('The average sales for Japan:\n',japan_sales_mean)

other_sales_total = df['other_sales_millions'].sum()
print('The total sales for Other:\n',other_sales_total)
other_sales_mean = df['other_sales_millions'].mean()
print('The average sales for Other:\n',other_sales_mean)

#steam dataset:

print('STEAM INFO:')
steam_counts=df_steam['steam'].value_counts()
print('The counts of if it is a steam game or not:\n',steam_counts)

steam_totals = df_steam.groupby('steam')['total_sales_millions']
print('The total sales depending on if it is a steam game or not:\n',steam_totals)

```
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/969e8bcb-e819-4c86-b1db-8d9cbe16c694)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/79d8be9e-85c2-437c-8440-7f3d26dacd34)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/cba71928-d070-4ebb-9510-8166fda20918)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/93c912b8-7ef1-4a5c-b852-5bb6dec2bc83)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/647bd843-b900-4e96-bbca-403bfb3c07b3)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/577fe15e-8573-43ae-83d7-93b36f88f001)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/cfe4e382-6fdd-4654-b222-03cb5b2dd214)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/87511f0d-f475-42d1-a443-a670b61199e4)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/efda74e7-5fbf-45a0-a633-c69318a08fda)

**Graphs**

```
    
sns.set_style("whitegrid", {'grid.linestyle': '--'})

#Graphs used in the final project
#graph 1a,b

sns.countplot(data=df,x='genre').set_title('Total Games Released per Genre')
plt.xticks(rotation=80)
plt.show()

sns.barplot(data=df,x='genre',y='total_sales_millions',estimator=np.mean).set_title('Average Sales per Genre')
plt.xticks(rotation=80)
plt.show()

```
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/f78c488a-ad66-480a-997d-0d95d310e3b8)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/d7883b88-9ae8-40ba-b503-bb5f3789369d)
```
#graph 2a,b

salesgrouping_order = ['1. Very Small','2. Small','3. Medium','4. Large','5. Very Large']
sns.countplot(data=df,x='console_manufacturer',hue='salesgrouping',hue_order=salesgrouping_order).set_title('Manufacturers Counts split by Sales Grouping')


sns.lineplot(data=df,x='release_year',y='total_sales_millions',ci=None,hue='console_manufacturer',palette=['blue','green','red','orange','deepskyblue','purple'],estimator=np.sum).set_title('Number of Games Sold per Year')

```
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/5ad78632-5467-47ff-a65c-08486fe505d3)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/89b391e1-b8fe-4806-bfce-f6c474b6a78f)

```
#graph 3a,b,c

sns.lineplot(data=df.groupby(['release_year']).size().reset_index(name='count'),x='release_year',y='count').set_title('Number of Games Released per Year')
plt.xticks([1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020],rotation=80)

sns.lineplot(data=df,x='release_year',y='total_sales_millions',estimator=np.mean).set_title('Average Sales per Year')
plt.xticks([1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020],rotation=80)
plt.show()

sns.lineplot(data=df,x='release_year',y='total_sales_millions',ci=None,estimator=np.sum).set_title('Total Sales per Year')
plt.xticks([1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020],rotation=80)
plt.show()
```
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/affc480d-fd5c-4692-9ce8-c19c708a6758)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/6e4bae19-8ea0-446a-a379-e2f0b7b1bf7e)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/d6feb00b-6ea8-4160-a025-efab764ea851)

```
#graph 4a,b

plt.scatter(df_scores['critic_score'], df_scores['user_score'])
plt.xlabel('Critic Score')
plt.ylabel('User Score')
plt.title('User vs Critic Scores')

plt.scatter(df_scores['critic_score'], df_scores['total_sales_millions'])
plt.xlabel('Critic Score')
plt.ylabel('Sales (millions)')
plt.title('Critic Scores vs Sales')

```
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/2c59ec84-b2e0-4523-8655-4713e915d2e9)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/c81d065d-40b6-498a-a61a-9caa52642399)
```
#Other graphs done but ultimately not used:

sns.countplot(data=df,x='console_manufacturer').set_title('Number of Games Release per Console Manufacturer')

sns.pairplot(data=df,vars=['genre','console_manufacturer','total_sales_millions'])

sns.kdeplot(data=df, x="release_year", hue="genre")
plt.xticks([1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020],rotation=80)
plt.show()

sns.kdeplot(data=df, x="release_year", hue="console_manufacturer").set_title('Games Released per Manufacturer')
plt.xticks([1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020],rotation=80)
plt.show()

sns.heatmap(df.corr(),cmap = 'coolwarm', annot=True)

```
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/0be75f5a-ee34-4a02-87a3-a2607620ea6e)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/4ed33f6e-15cb-4748-8933-7591a29a5a18)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/d0f4e0a0-83ae-4cb1-92e5-d3b363460af4)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/0a07ee7d-f715-4bec-b500-85fa9f4e6a56)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/d704fd5a-4e31-4800-9b85-2a1b8bb2f833)
**Finding Outliers**
```
print('The mean and standard deviation for the total sold:\n',df['total_sales_millions'].agg(['mean','std']))
total_sales_mean = df['total_sales_millions'].mean()
print('The mean for the total sales:',total_sales_mean)
total_sales_std = df['total_sales_millions'].std()
print('The std for the total sales:',total_sales_std)
total_sales_75th = df['total_sales_millions'].quantile(0.75)
total_sales_25th = df['total_sales_millions'].quantile(0.25)
total_sales_iqr = total_sales_75th - total_sales_25th
print('The 25th and 75th percentiles of the total sales are: ',total_sales_25th,' and ',total_sales_75th,' respectively. The interquartile range is ',total_sales_iqr)
total_sales_upper = total_sales_75th + (1.5 * total_sales_iqr)
total_sales_lower = total_sales_25th - (1.5 * total_sales_iqr)
print('The total sales upper is: ',total_sales_upper,' and the lower is: ',total_sales_lower)

total_sales_upper_outliers = df[(df['total_sales_millions'])>total_sales_upper]
print(total_sales_upper_outliers)
total_sales_lower_outliers = df[(df['total_sales_millions'])<total_sales_lower]
print(total_sales_lower_outliers) # no lower as the lower cut off is a negative number and we can't have negative sales

#The upper outliers appear to be valid - the most extreme is GTA V and when I checked the total sales, according to all sites I could see, this seems to be a valid number


no_outliers = df[df['total_sales_millions']<total_sales_upper ] 
#no need to inlcude removing lower outliers as there are none.

print(df.shape)
print(no_outliers.shape)
print(df.head())
print(no_outliers.head())
print(df.info())
print(no_outliers.info())
print(df.describe())
print(no_outliers.describe())
```
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/97c312f9-b621-40a3-aa3d-22f5ceed2d67)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/c751e6c0-912c-4abf-958c-c3dfae0ee55d)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/de2d7b3d-116b-40a5-8f5f-c6158107e8fc)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/5b79677e-d455-400f-b685-5b4d54df6c79)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/b7749972-1419-42ba-a85f-c17eeb77a7cf)

**Further Statistical Analysis**
```
ecdf_x = np.sort(df['total_sales_millions'])
ecdf_y = np.arange(1,len(ecdf_x)+1)/len(ecdf_x)
_ = plt.plot(ecdf_x,ecdf_y,marker='.',linestyle='none')
_ = plt.xlabel('Total Sales (millions)')
_ = plt.ylabel('EDCF')


plt.margins(0.02)
plt.show()



#sales exponential distribution
total_sales_exp = np.random.exponential(total_sales_mean,10000)

_ = plt.hist(total_sales_exp, histtype='step', bins=50)
_ = plt.xlabel('Total Sales (millions)')
_ = plt.ylabel('PDF')
plt.show()

total_sales_percentiles = np.percentile(df['total_sales_millions'],[25,50,75])

_ = sns.boxplot(x='console_manufacturer',y='total_sales_millions',data=df)
_ = plt.xlabel('Manufacturers')
_ = plt.ylabel('Total Sales (millions)')
plt.show()

total_sales_var = np.var(df['total_sales_millions'])
total_sales_sqrt = np.sqrt(total_sales_var)

```
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/6aaea392-5d6b-4e9b-8726-f7177230aa92)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/5799488c-eed4-4b3b-bf58-b687255dbb2f)

