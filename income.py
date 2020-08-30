def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#Start by loading data into a pandas dataframe, taking out spaces and making the first inputs the header
income_data = pd.read_csv('income.csv', header = 0, delimiter=", ")
#print(income_data.iloc[0])

#Create labels for the random forest
labels = income_data[['income']]

#Create new column for sex that features integers for decision tree
income_data['sex-int'] = income_data['sex'].apply(lambda row: 0 if row == 'Male' else 1)

#Create a new column for countries, since the United States makes up most of the country data
income_data['country-int'] = income_data['native-country'].apply(lambda row: 0 if row == 'United-States' else 1)

#Create data, using a variety of columns
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int']]

#Create training and testing data subsets for model
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

#Create, and fit, the RFC model
forest = RandomForestClassifier(random_state=1)

forest.fit(train_data, train_labels)

#Showing which features are the most influencial: age is the primary features, capiltal gain is second, and hours per week is third
print(forest.feature_importances_)

#Model scores at an 82.25% accuracy rating
print(forest.score(test_data, test_labels))





