import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.ticker as tck
import matplotlib.ticker as pltck
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 

#importing the dataset
ipl = pd.read_csv('E:\Only-CO-Lang\Python Code\CODE-py\cricket\ipl.csv')
print(f"Dataset successfully Imported of Shape : {ipl.shape}")

ipl = pd.read_csv('E:\Only-CO-Lang\Python Code\CODE-py\cricket\ipl.csv')
print(ipl)

##DATA ANALYSIS
ipl.head(100)

ipl.describe()
ipl.info()
ipl.dtypes

#distribution of wickts
sns.displot(ipl['wickets'],kde=False,bins=10)
plt.title("Wicket Distribution")
plt.show()

#distribution of runs
sns.displot(ipl['total'],kde=False,bins=10)
plt.title("Run Distribution")
plt.show()
ipl.columns

irrelevant = ['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
print(f"Before removing irrelevant columns: {ipl.shape}")
ipl = ipl.drop(irrelevant, axis=1)
print(f"After removing irrelevant columns: {ipl.shape}")
ipl.head()

const_team=['Kolkata knight Riders','Chennai Super Kings','Rajathan Royals','Mumbai Indians','Kings XI Punjab','Royal Challengers Banglore','Delhi Daredevils','Sunrisers Hyderabad']

#define consistent teams
print(f'Before Removing Inconsistent Teams : {ipl.shape}')
ipl = ipl[(ipl['bat_team'].isin(const_team)) & (ipl['bowl_team'].isin(const_team))]
print(f'After Removing Irrelevant Columns : {ipl.shape}')
print(f"Consistent Teams : \n{ipl['bat_team'].unique()}")
ipl.head()

print(f'Before Removing Overs : {ipl.shape}')
ipl = ipl[ipl['overs'] >= 5.0]
print(f'After Removing Overs : {ipl.shape}')
ipl.head()

from seaborn import heatmap
heatmap(data=ipl.corr(numeric_only=True), annot=True)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
for col in ['bat_team', 'bowl_team']:
    ipl[col] = le.fit_transform(ipl[col])
ipl.head()

from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[0, 1])],remainder='passthrough')
ipl=np.array(columnTransformer.fit_transform(ipl))
cols = ['bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals', 'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad', 'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab','bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals','bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']
df = pd.DataFrame(ipl)


