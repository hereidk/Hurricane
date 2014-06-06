'''
Created on Apr 10, 2013

@author: kahere
'''

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import xlrd
# import scipy.io as sio
# import matplotlib.pyplot as plt
# import cProfile
# import itertools
# import exceptions

if __name__ == '__main__':
    Proxy = xlrd.open_workbook(r'C:\Python code\Hurricane\src\root\nested\HurrData.xlsx')
    Proxy_data = Proxy.sheet_by_name('Data')
    values = np.zeros((Proxy_data.nrows,Proxy_data.ncols))
    for row in range(Proxy_data.nrows):
        for col in range(Proxy_data.ncols):
            values[row,col] = (Proxy_data.cell(row,col).value)
            
    tests = values[:-1,1:5]
    predictors = values[:-1,5:]
    
    predict_2014 = values[-1,5:]
    
    # Reshape datasets so they are all [years, data type]
    predictors = np.reshape(predictors,[np.int(np.size(predictors)/np.size(predictors[0])),np.size(predictors[0])])
    tests = np.reshape(tests,[np.int(np.size(tests)/np.size(tests[0])),np.size(tests[0])])
    predict_2014 = np.reshape(predict_2014,[1,np.size(predict_2014)])

    # Make prediction for (in order) number of named storms, hurricanes, major hurricanes, ACE index
    for i in range(0,4):
        CForest = RandomForestRegressor(n_estimators = 10000)
        CForest = CForest.fit(predictors,tests[:,i])
        storm_predict = CForest.predict(predict_2014)
    
        print (storm_predict[0])
     
    # Remove one year at a time, make hindcasts
    validation_set = np.zeros([np.size(predictors,0),4])   
    for j in range(0,np.size(predictors,0)):
        training = np.delete(predictors,j,0)    
        match = np.delete(tests,j,0)         
        length = np.size(match,0)
    
        training_year = predictors[j,:]
        training_year = np.reshape(training_year,[1,np.size(training_year)])
        
        CForest = RandomForestRegressor(n_estimators = 10000)
        CForest = CForest.fit(training,match)
        validation_set[j,:] = CForest.predict(training_year)
        
    print (validation_set)
    
    
    