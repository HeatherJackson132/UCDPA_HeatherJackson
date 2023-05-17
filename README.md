# UCD_DataAnalytics_VGChartz

Project Report
GitHub URL
https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz

Contents:
 - webscrape.py
 - data_analytics_essentials.py
 - RawData.csv
 - steam.csv
 - RawFiles – folder containing 20 raw csv files that make up the RawData csv
 - Project Notes.md


**Abstract**

The goal of this project is to investigate the predicted sales on a video game based on any particular genre or console or year of release. The dataset was taken from a website called VG Chartz which provides video game sales. This website collates publicly available sales figures for games going back as far as 1970. A second dataset was taken from a Kaggle dataset of games available on Steam to investigate the effect of publishing a computer game on Steam or not.

**Introduction**

I chose a video game dataset as video games are becoming more and more popular. They are no longer only for children but for all ages. Video games make more now than music and movies according to a report by SuperData Research. I myself have played games from a young age, beginning with Sonic on my older brother’s Sega Mega Drive and I was interested to see how the industry and evolved and adapted. 

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

**Implementation Process**

Spyder was used to import and analyse the data then perform the machine learning. I did an initial overview of the data. Missing data and unneeded data was removed. Where there was 0 sales but not null, this was replaced with 0.0025 million (as the data was in the format x.xx so 0 would be between 0 and 0.0045 approximately so I took the middle of the values that it could have been).

3 datasets were created from the original 2:

 - Dataset 1 – df – The same as the original data but with the null total sales value rows removed and the columns updated
 - Dataset 2 – df_steam – The computer data only merged with the Steam dataset
 - Dataset 3 – df_scores – The same as the original but with null ratings removed

I also created a dataset called no_outliers which had the outliers removed. However I would consider the outliers viable data in this case as when I checked them, they were all very high values but according to google, the values are realistic. I decided to keep the outliers in while reviewing the data and then to do the machine learning on both to see if removing the outliers had a significant effect on the accuracy.

I choose 4 methods of machine learning and compared the values with and without the outliers. I then used Randomized Search CV method of hyperparameter tuning

**Results**
 
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/e5f1a504-8b2b-4bee-902f-05080bc33512)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/e6fc6aa1-01cb-452c-81f3-cade23e78768)

Graph 1: The games released and average sales for each genre
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/e014aee4-d7b1-43eb-bfeb-dd7514f95168)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/ad26c701-1be3-4fa8-a6bc-6ea53f665d90)

Graph 2: Counts of games for each manufactorer split by sales grouping
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/ef13e58f-c1eb-458c-97bc-3e57ce08fa82)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/4ef75695-e55d-4dd1-8ba9-ba810434a0d9)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/0ea5b8f2-3512-4c9d-b412-1610ac2dca58)

Graph 3: Set of 3 comparing number of games against average sales per game and total sales.
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/be3f89d1-604e-4dfb-80fc-d3011bd05532)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/f86be3e6-b4d8-4faa-9cf7-4071394611c2)
Graph 4: Comparing Critic Score against User Score and Sales
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/aa65ef25-dc30-45fa-9bde-f5649c26e636)
![image](https://github.com/HeatherJackson132/UCD_DataAnalytics_VGChartz/assets/133404925/507b97b4-0825-466e-a652-34be68c56728)

Graph 5: Comparing sales if game is on steam or not

I did machine learning to predict how many sales a game would have depending on the year it was brought out. 

Model	                        Accuracy when data includes outliers  Accuracy when data excludes outliers
Decision Tree classifier      0.301763668	                          0.333791749
Bagging Classifier	          0.299823633	                          0.333005894
Logistic Regression	          0.305996473	                          0.33654224
Gradient Boosting Classifier	0.303703704	                          0.334774067

Despite having a high number of records, the accuracy of all models is quiet low, even with the outliers removed.

With and without the outliers, the best model by a fraction was the Logistic Regression model.

I did the Randomized Search CV method of hyperparameter tuning however the accuracy of Randomized Search CV was only 0.3347677111476783.

I believe that models for predicting data or improving the predictions were not accurate because there was too much variation in the data. 

Insights

 - Genre Insights: 
 - Action and Sports games were 2 of the genres with the most games released, followed closely by Adventure and Misc. However, the average sales of Action and Sports were middle of the road. While Adventure had one of the lower average sales. As Action and Sports have so many games made, it is likely because companies believe they can sell a significant number of copies. In fact, when I look at the data, they are indeed some of the highest selling games and the top 10 action or sports games actually average nearly 14 million between them. For them to be in the middle for average sales, this must mean that while a small number sell a lot, there must be a large number that do not. I believe that the small number of high selling games are unicorns and the rest of the games are trying to capture that magic. But in doing so may have flooded the market so that the group interested in them are too spread out. 
 - The sandbox games are the opposite. There is only one game that did reasonably well and because there were no poorly received games to drag the genre down, it has the highest average sales by a significant margin.

 - Manufacturer Insights:
 - The consoles that sold the highest number of games are Nintendo and Sony. Nintendo had a rapid incline in the early 2000s that flatten before jumping again, peaking at around 2010 followed by a sharp decline. I actually found this quite unexpected as I would have thought the N64 era was Nintendo at the height of their power. But this could be explained by gaming at the time being targeted more toward children only and the prevalence of the rental market – so while they were the highest sellers at the time, it was a significantly smaller market. I would think that the success in the late 2000s and early 2010s would have been down to their success with the DS/3DS as well as the Wii which were unique amongst the consoles available at the time so cornered a different area of the market. However, with the release of the WiiU, the public lost a lot of faith in them. It does start to recover in the late 2010s with the release of the Switch.
 - Sony had a more consistent rise and domination over Microsoft who have been their rival since the release of the Xbox. There were  obvious peaks around the year 2000, in the late 2000s and in the mid 2010s. These coincide with the releases of the PlayStation 2 in 2000, PlayStation 3 in 2006 and PlayStation 4 in 2013. There is a similar peaks for Microsoft but they never reach the same levels.

 - Games per Year Insights:
 - These results were the most unexpected – I had expected there to be an overall trend upwards as the video game industry gain popularity with some variation due to environmental factors. However, the peaks were very unexpected with the highest number of games being released and sold was in the early 2000s to the mid 2010s. Though, this is also where there is the lowest average sales per game implying that there was a certain amount of market saturation at this point. This is made more obvious be the fact that the highest average sales per game were between the 90s and 80s where they also had the lowest number of games released. So people had limited options. I had expected there to be a drop around 2009 for the recession as they were a luxury rather than a requirement but as mentioned above, this could be down to the popularity of the consoles and games released at that time. 
 - There is a sharp decline in the late 2010s and 2020 however this is expected as the sales figures are per game for all time so the most recently released games haven’t had as much time to be purchased.

- User and Critic Scores Insights:
 - There was a general correlation between the users and critics, there were some outliers however they are quite consistent.
 - The sales are more interesting with the sales varying wildly. Games with a lower critic scores do not tend to sell well however once the critics score reached 6.5, the  sales could stay as low as a poorly reviewed game or be significantly higher than a game that was better reviewed.

 - Steam Insights:
 - There was a significantly larger portion of games not on Steam than there were on it. However this could be down to how they were linked. Not all games will be named the same between the 2 data sets. This is particularly true for sequels where, eg Age of Empires III The Asian Dynasties could have been Age of Empires 3 or III with or without Asian Dynasties. I removed special characters to at least stop things like : or – from causing mismatches
 - Despite the fact that there were only 30% of games on Steam, on average, if a game was on PC, the sales were higher if it was on Steam. This would make sense as Steam is a widely used platform that makes games easily accessible. This is the case when I reviewed the games made by the various EA developers who have a PC platform called Origin. The average sales per game were double if the game was on Steam. 

**References**

Web scraping references:
•	https://www.vgchartz.com/methodology.php
•	https://www.kaggle.com/datasets/nikdavis/steam-store-games
•	https://github.com/jeremyrchow/video_game_sales_data/blob/master/scraping_vgchartz.py
