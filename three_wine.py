# 3 Different Classifiers on Wine Data

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

# Read data
df = pd.read_csv('three_wine.csv')
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
        
# Train-Test split
x = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
y = df['color (0 = red, 1 = white)'].values
        
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)

# K-NN
wine_knn = KNeighborsClassifier(n_neighbors = 5).fit(x_train,y_train) # k = 5
y_hat = wine_knn.predict(x_test)

print("K-NN Accuracy: ", metrics.accuracy_score(y_test, y_hat))

# Decision Tree
wine_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 5)
wine_tree.fit(x_train, y_train)
y_hat = wine_tree.predict(x_test)

print("Decision Tree Accuracy: ", metrics.accuracy_score(y_test, y_hat))

# Support Vector Machine
wine_svm = svm.SVC(kernel='rbf')
wine_svm.fit(x_train, y_train)
y_hat = wine_svm.predict(x_test)

print("SVM Accuracy: ", metrics.accuracy_score(y_test, y_hat))
