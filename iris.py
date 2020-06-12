# Iris flower classifier

# Imports
import math
import pandas as pd
from sklearn.model_selection import KFold
from scipy import stats as stats
from statistics import mean

# Read data
df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Projects_S2020\\edX\\ML\iris.csv')
df['guess class'] = ''

# Normalize Data
for col in df.columns:
    
    if (col == 'class' or col == 'guess class'):
        pass
    
    else:    
        m = df[col].mean()
        std = df[col].std()
        
        df[col] = (df[col] - m)/std
        
# Euclidean distance function
def Euc(x,y):
    """Determines the Euclidean distance between x and y,
    where x and y are rows of dataframe
    
    x = np.array(train.iloc[index])"""
    
    dist = 0
    i = 0
    
    while (i < x.shape[0] - 2):
        dist += (x[i] - y[i])**2
        i += 1
    
    dist = math.sqrt(dist)
    
    return dist

# K-NN Algorithm
def KNN(train,test,folds,k):
    """Trains k-nn algorithm on train set and tests it on test set, 
    returns test set with guesses and accuracy"""
    
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
        
        while (m < k):
            neb.append(train.iloc[dist_list.index(s_dist_list[m])]['class'])
            m += 1
        
        guess = stats.mode(neb)[0]
        test.at[i, 'guess class'] = guess
        #test.iloc[i]['guess class'] = guess
            
        i += 1
    
    # Evaluate
    intersection = []
    i = 0
    
    while (i < test.shape[0]/folds): 
        
        if (test.iloc[i]['class'] == test.iloc[i]['guess class']):
            intersection.append('a')
        
        else:
            pass
        
        i += 1
    
    S = len(intersection)/(test.shape[0]/folds)
    
    return test, S;

# Perform K-fold cross validation
kf = KFold(n_splits = 5, shuffle = True)
folds = kf.split(df)

score_list = []


for fold in folds:
    test = df.iloc[fold[0]]
    train = df.iloc[fold[1]]
    
    test, S = KNN(train,test,kf.get_n_splits(),3) # final input is k in knn
    
    score_list.append(S)
    
print(f'Score: {mean(score_list)}')
print(score_list)
