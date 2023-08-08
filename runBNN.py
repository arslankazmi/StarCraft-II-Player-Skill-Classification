from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import CSVLogger
from keras.utils import np_utils
from keras import optimizers
from keras import layers
from keras.models import load_model
import numpy
import pandas as pd
import os


if os.path.exists("history.csv"):
	os.remove("history.csv")#delete old history file


# fix random seed for reproducibility
numpy.random.seed(7)

# Defining column names for dataset
X_COLUMN_NAMES = ['APM','SelectByHotkeys','AssignToHotkeys','UniqueHotkeys','MinimapAttacks','MinimapRightClicks','NumberOfPACs','GapBetweenPACs','ActionLatency','ActionsInPAC','TotalMapExplored','WorkersMade','UniqueUnitsMade','ComplexUnitsMade','ComplexAbilitiesUsed']
Y_COLUMN_NAMES = ['LeagueIndex']

# Import training dataset
#training_dataset_X = pd.read_csv('SkillCraft1_Dataset_modified.csv', names=X_COLUMN_NAMES, header=None)#using whole dataset forcross validation
training_dataset_X = pd.read_csv('trainX.csv', names=X_COLUMN_NAMES, header=None)
print(training_dataset_X.head())

train_x = training_dataset_X.iloc[:,:].values



training_dataset_Y = pd.read_csv('trainY.csv', names=Y_COLUMN_NAMES, header=None)
training_dataset_Y.loc[:,'LeagueIndex'] = training_dataset_Y.loc[:,'LeagueIndex'] - 1;
print("First rows of Training dataset Y: \n\n",training_dataset_Y.head(),"\n*********************************\n")
print("Training Dataset Y summary: \n\n",training_dataset_Y.describe(),"\n*********************************\n")


train_y = training_dataset_Y.iloc[:,:].values

# Encoding training dataset
encoding_train_y = np_utils.to_categorical(train_y)

# Import testing dataset
test_dataset_X = pd.read_csv('testX.csv', names=X_COLUMN_NAMES, header=None)
test_x = test_dataset_X.iloc[:, :].values

test_dataset_Y = pd.read_csv('testY.csv', names=Y_COLUMN_NAMES, header=None)
test_dataset_Y.loc[:,'LeagueIndex'] = test_dataset_Y.loc[:,'LeagueIndex'] - 1;

test_y = test_dataset_Y.iloc[:, :].values

# Encoding training dataset
encoding_test_y = np_utils.to_categorical(test_y)

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

csv_logger = CSVLogger('history.csv', append=True, separator=',')

# Training a model
history = model.fit(train_x, encoding_train_y, epochs=100, batch_size=60, validation_split = 0.25,callbacks=[csv_logger])

model.save('arslan.h5');

#model.fit(train_x, train_y, epochs=50, batch_size=10)

fileModel = load_model('arslan.h5');

# evaluate the model
scores = fileModel.evaluate(test_x, encoding_test_y)
#scores = model.evaluate(test_x, test_y)

print("\nAccuracy: %.2f%%" % (scores[1]*100))

#model_predictions = fileModel.predict(test_x);

#print("\n\n",model_predictions[0:5])

