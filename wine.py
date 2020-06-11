# Wine data classification

# Imports
import pandas as pd
import numpy as np
import math

# Read data
df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Projects_S2020\\edX\\ML\wine.csv')
df.drop(['quality'], axis=1, inplace=True)
df['color (0 = red, 1 = white)'] = df['color (0 = red, 1 = white)'].astype(float)

# Normalize Data
for col in df.columns:
    
    if (col == 'color (0 = red, 1 = white)'):
        pass
    
    else:    
        m = df[col].mean()
        std = df[col].std()
        
        df[col] = (df[col] - m)/std

# Split data into train and test sets
mask = np.random.rand(len(df)) < 0.8
train = df[mask]
test = df[~mask]
test.reset_index(drop=True)

def Euc(x,y):
    """Determines the Euclidean distance between x and y,
    where x and y are rows of dataframe
    
    x = np.array(train.iloc[index])"""
    
    dist = 0
    i = 0
    
    while (i < 11):
        dist += (x[i] - y[i])**2
        i += 1
    
    dist = math.sqrt(dist)
    
    return dist

# K-NN Algorithm
i = 0

while (i < test.shape[0]): 
    x = test.iloc[i]
    dist_list = []
    neb = []
    
    j = 0
    
    while (j < train.shape[0]):
        y = train.iloc[j]
        dist_list.append(Euc(x,y))
        j += 1
       
    s_dist_list = sorted(dist_list)
    
    m = 0
    n = 5 # n here is the K in K-NN
    
    while (m < n):
        neb.append(train.iloc[dist_list.index(s_dist_list[m])]['color (0 = red, 1 = white)'])
        m += 1
    
    neb_sum = sum(neb)
    
    if (neb_sum >= 3):
        test.iloc[i]['guess'] = 1
        
    else:
        test.iloc[i]['guess'] = 0
        
    i += 1

# Evaluate
intersection = []
i = 0

while (i < test.shape[0]): 
    
    if (test.iloc[i]['color (0 = red, 1 = white)'] == test.iloc[i]['guess']):
        intersection.append('a')
    
    else:
        pass
    
    i += 1
    
J = len(intersection)/(2*(test.shape[0]) - len(intersection))
print(f'Jaccard Index: {J}')

test.to_csv('wine_output.csv')

# Tested to Jaccard Index of 0.9883466860888566
