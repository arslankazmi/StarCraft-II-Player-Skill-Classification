

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy
import pandas as pd
import os


if os.path.exists("history.csv"):
	os.remove("history.csv")#delete old history file




# fix random seed for reproducibility
numpy.random.seed(7)


#missing values are question marks in this data set
missingValues = ["?"]

#read csv into dataframe 
df = pd.read_csv("SkillCraft1_Dataset_modified.csv", na_values = missingValues)

print("*****************************************\nCorrelation coefficients: \n")

print(df.corr())


print("****************************************\n")

print(list(df))

print("****************************************\n")

print("Getting X and Y arrays...")

X = df[['APM','SelectByHotkeys','AssignToHotkeys','UniqueHotkeys','MinimapAttacks','MinimapRightClicks','NumberOfPACs','GapBetweenPACs','ActionLatency','ActionsInPAC','TotalMapExplored','WorkersMade','UniqueUnitsMade','ComplexUnitsMade','ComplexAbilitiesUsed']]

Y = df['LeagueIndex']

print("Splitting into Training and Testing arrays....")

trainX , testX , trainY, testY = train_test_split(X , Y , test_size=0.2)

print("Outputting trainging and testing datasets to files...")

with open('trainX.csv', 'w') as FOUT:
    numpy.savetxt(FOUT, trainX, delimiter = ',')

with open('testX.csv', 'w') as FOUT:
    numpy.savetxt(FOUT, testX, delimiter = ',')

with open('trainY.csv', 'w') as FOUT:
    numpy.savetxt(FOUT, trainY, delimiter = ',')

with open('testY.csv', 'w') as FOUT:
    numpy.savetxt(FOUT, testY, delimiter = ',')

print("Creating and Training model...")

lm = LinearRegression()

model = lm.fit(trainX , trainY)


print("Visualizing predictions...")

predictions = lm.predict(testX)

with open('predictionsMLR.txt', 'w') as FOUT:
    numpy.savetxt(FOUT, predictions,delimiter = ',')





