import numpy as np
import pandas as pd

def preprocess(data):
    DF = data[['Match','Innings', 'Over', 'Total.Runs', 'Innings.Total.Runs', 'Wickets.in.Hand']].copy()   
    DF = DF[DF['Innings']==1].reset_index()    #Considering only 1st innings

    #Advanced Pre-processing (Removing incomplete first innings)
    Total_Matches_played = DF[DF['Total.Runs'] == DF['Innings.Total.Runs']].reset_index()
    Incomplete_innings = Total_Matches_played[(Total_Matches_played['Over'] != 50) & (Total_Matches_played['Wickets.in.Hand'] != 0)]
    Incomplete_matchNum = Incomplete_innings['Match']
    Incomplete_matchNum = Incomplete_matchNum.to_numpy(dtype = np.int64)

    train_set = DF
    for i in range(np.size(Incomplete_matchNum)):
        train_set = train_set[train_set['Match'] != Incomplete_matchNum[i]]
    
    return train_set