import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split




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
    np.savetxt(FOUT, trainX, delimiter = ',')

with open('testX.csv', 'w') as FOUT:
    np.savetxt(FOUT, testX, delimiter = ',')

with open('trainY.csv', 'w') as FOUT:
    np.savetxt(FOUT, trainY, delimiter = ',')

with open('testY.csv', 'w') as FOUT:
    np.savetxt(FOUT, testY, delimiter = ',')

print("Creating and Training model...")

lm = LinearRegression()

model = lm.fit(trainX , trainY)


print("Visualizing predictions...")

predictions = lm.predict(testX)

with open('predictionsMLR.txt', 'w') as FOUT:
    np.savetxt(FOUT, predictions,delimiter = ',')

plt.scatter(testY,predictions)



print("Score: ", model.score(testX, testY))

plt.title('SkillCraft1 Multiple Linear Regression Predictions')
plt.xlabel('Expected LeagueIndex')
plt.ylabel('Predicted LeagueIndex')
plt.show()


