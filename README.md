# EXPLORATORY-DATA-ANALYSIS
Explored and preprocessed the Titanic dataset. Visualized survival patterns by class, sex, and age. Handled missing values, converted categorical features to numeric, and cleaned the data. Built a logistic regression model to predict survival, showcasing data-driven insights and basic machine learning.

 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv(r"C:\Users\GOKUL\OneDrive\Desktop\Skill up\ML\titanic_train.csv")
# first we are finding the missing values
print(train.isnull())
# Now we use a Seaborn a (Data Visualization Library) because it shows missing values in a coloured grid(1)[Heatmap]
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='coolwarm')
plt.title("Missing Values Heatmap")
plt.show()
# Checking how many missing data in the each column
# Basic plot: Survival count
sns.set_style('whitegrid')
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=train, palette='Set2')
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()
# 3. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=train, palette='Set1')
plt.title("Survival by Passenger Class")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()
# 4. Survival by Sex
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=train, palette='Set2')
plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()
# 5. Age distribution of survivors
plt.figure(figsize=(8,6))
sns.histplot(data=train, x='Age', hue='Survived', multiple='stack', kde=False, palette='Set1')
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Okay, Let's do the-- Data Cleaning-- and Fill in the missing values

#step 1 : Performing boxplot to visualize the distribution of AGE across various passenger class (6)

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
plt.title("Box Plot")
plt.show()

#step 2 : Fill in the missing values(7)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
# drop an cabin column (because it has more than 80 percent missing values)
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.title("New Heatmap")
plt.show()

## CONVERTING CATEGORICAL FEATURES (text to numeric conversion)
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embark], axis=1)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
print("Cleaned dataset preview:\n", train.head())
print("\nDataset info:\n")
train.info()

## STEPS TO BUILD A LOGISTIC REGRESSION

# Features and target
X = train.drop('Survived', axis=1)
y = train['Survived']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train logistic regression model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='liblinear')  # works well for small datasets
logmodel.fit(X_train, y_train)

# Make predictions on the test set
predictions = logmodel.predict(X_test)

# Evaluate model
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))                                                  #------Done by S.Gokul-------------#
