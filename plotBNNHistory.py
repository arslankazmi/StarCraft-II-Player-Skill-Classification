import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#read csv into dataframe 
hist = pd.read_csv("history1.csv")

#ax = plt.gca()

#hist.plot(kind='line',x='epoch',y='acc',ax=ax)

plt.figure(1)  
   
# summarize history for accuracy  

plt.subplot(211)  
plt.plot(hist.iloc[:,0].values,hist.iloc[:,1].values)  
plt.plot(hist.iloc[:,0].values,hist.iloc[:,3].values)  
plt.title('Model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  

# summarize history for loss  

plt.subplot(212)  
plt.plot(hist.iloc[:,0].values,hist.iloc[:,2].values)  
plt.plot(hist.iloc[:,0].values,hist.iloc[:,4].values)  
plt.title('Model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  


plt.show()  

