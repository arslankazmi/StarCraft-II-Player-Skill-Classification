from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import CSVLogger
from keras.utils import np_utils
from keras import optimizers
from keras import layers
from sklearn.cross_validation import StratifiedKFold
import numpy
import pandas as pd
import os




#missing values are question marks in this data set
missingValues = ["?"]

#read csv into dataframe 
df = pd.read_csv("SkillCraft1_Dataset_modified.csv", na_values = missingValues)

df.loc[:,'LeagueIndex'] = df.loc[:,'LeagueIndex'] - 1;

# fix random seed for reproducibility
numpy.random.seed(7)

# Defining column names for dataset
X_COLUMN_NAMES = ['APM','SelectByHotkeys','AssignToHotkeys','UniqueHotkeys','MinimapAttacks','MinimapRightClicks','NumberOfPACs','GapBetweenPACs','ActionLatency','ActionsInPAC','TotalMapExplored','WorkersMade','UniqueUnitsMade','ComplexUnitsMade','ComplexAbilitiesUsed']
Y_COLUMN_NAMES = ['LeagueIndex']


total_x = df.iloc[:,1:16].values
total_y = df.iloc[:,0].values

print(total_y)
#print(total_x)


n_folds = 10
#skf = StratifiedKFold(n_splits=n_folds)
#skf.get_n_splits(total_x, total_y)

labels = [0,1,2,3,4,5,6,7]

skf = StratifiedKFold(total_y, n_folds=n_folds, shuffle=True)

sumOfAccs = 0;

for i, (train, test) in enumerate(skf):

	train_x = total_x[train]
	test_x = total_x[test]

	# Encoding training dataset
	encoding_train_y = np_utils.to_categorical(total_y[train])

	# Encoding training dataset
	encoding_test_y = np_utils.to_categorical(total_y[test])

	activation = layers.LeakyReLU(alpha=0.2)

	# Creating a model
	model = Sequential()
	model.add(Dense(15, input_dim=15))
	model.add(Dense(35, activation='sigmoid'))
	#model.add(Dense(1, activation='sigmoid'))
	model.add(Dense(8, activation='softmax')) # when league indices have been converted to 0-7 instead of 1-8
	#model.add(Dense(9, activation='softmax'))
	#model.add(Dense(1, activation='linear')) #for non-softmax andnon-encoded to categorical



	print(model.summary())


	#Choosing optimizer
	sgdOptimizer = optimizers.SGD(lr=0.0025, momentum=0.2, decay=0.0, nesterov=False)

	# Compiling model
	#model.compile(loss='categorical_crossentropy', optimizer = sgdOptimizer, metrics=['accuracy'])
	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

	csv_logger = CSVLogger('history' + str(i) + '.csv', append=True, separator=',')

	# Training a model
	history = model.fit(train_x, encoding_train_y, epochs=100, batch_size=60, validation_split = 0.25,callbacks=[csv_logger])


	#model.fit(train_x, train_y, epochs=50, batch_size=10)

	# evaluate the model
	scores = model.evaluate(test_x, encoding_test_y)
	#scores = model.evaluate(test_x, test_y)

	sumOfAccs += scores[1];

	print("\nAccuracy: %.2f%%" % (scores[1]*100))


print("\nAccuracy overall: %.2f%%" % ((sumOfAccs / n_folds)*100))
