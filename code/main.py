from clear_duplicate import cd
from bert import judge_emotion
from seperate_time import time
from countnum import count
from lat_lon import latlon
from comments import comment
from graph import g
from cluster import main1
from tree_v2 import qt

#Cleaning up duplicate data
cd('../data/wh_data.csv','../data/wh_data_cleaned.csv')
#Sentiment analysis
judge_emotion('../data/wh_data_cleaned.csv')
count('../data/emotion_prediction_wh.csv')
#revise the size according to your need
file='../result/bert/wh/256'
size=256
#Gridding of areas
latlon('../data/emotion_prediction_wh.csv',file,size)
#Placement of comment data into the corresponding grid
comment(file,'../data/emotion_prediction_wh.csv')
#Mood mapping
g(file)
#similarity clustering
main1(file)
#quadtree construction
qt(file,size)