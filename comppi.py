# -*- coding: utf-8 -*-
"""

Miro Tuokko, Joni Salmi, Kalle Mattila
Group 43
SGN-41007 Pattern recognition and Machine Learning

"""

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Conv1D, LSTM, Dropout, MaxPooling1D
from keras.utils import to_categorical
from keras import regularizers


if __name__ == '__main__':
    #read the data
    X_kaggle = np.load('X_test_kaggle.npy')
    Xtrain = np.load('X_train_kaggle.npy')
    X1, groups, X3  = np.loadtxt('groups.csv', dtype={'names':('Id', 'Group Id', 'Surface'), 'formats':(np.int, np.int, np.str)}, delimiter=',', unpack='true', skiprows=0)
    surfs = np.loadtxt('groups.csv', dtype=np.str, delimiter=',', usecols=[2])
    

    le = LabelEncoder()
    labelsnum = le.fit_transform(surfs)
    
    #Delete orientation data
    #Xtrain = np.delete(Xtrain, (0,1,2,3), axis=1)
    
    
    shuffler = GroupShuffleSplit(test_size=0.20, random_state=42)
    
    y = to_categorical(labelsnum, num_classes=9)
   
    for train, test in shuffler.split(Xtrain, labelsnum, groups):
        #print(train)
        #print(test)
        X_tr = Xtrain[train]
        y_tr = y[train]
        
        X_te = Xtrain[test]
        y_te = y[test]
        y_te2 = labelsnum[test]
        
    X_tr_means = np.mean(X_tr, axis=2)
    X_te_means = np.mean(X_te, axis=2)
        
    X_tr_std = np.std(X_tr, axis=2)
    X_te_std = np.std(X_te, axis=2)

    X_tr_mstd = np.expand_dims(X_tr_means, axis=2)
    X_tr_mstd = np.insert(X_tr_mstd, 0, X_tr_std, axis=2)

    X_te_mstd = np.expand_dims(X_te_means, axis=2)
    X_te_mstd = np.insert(X_te_mstd, 0, X_te_std, axis=2)
    
    X_tr_resize = np.resize(X_tr, (X_tr.shape[0], 1280))
    X_te_resize = np.resize(X_te, (X_te.shape[0], 1280))
    print(X_tr[:,0].shape)
    print(X_tr_means[:,0].shape)
    X_tr_all = np.insert(X_tr, 128, X_tr_means, axis=2)
    X_te_all = np.insert(X_te, 128, X_te_means, axis=2) 
    
    X_tr_all = np.insert(X_tr_all, 129, X_tr_std, axis=2)
    X_te_all = np.insert(X_te_all, 129, X_te_std, axis=2)
    
    
    #X_tr_all_1 = np.resize(X_tr_all, (X_tr_all.shape[0], 10, 130))
    X_tr_all = np.transpose(X_tr_all, axes=[0, 2, 1])
    X_te_all = np.transpose(X_te_all, axes=[0, 2, 1])

        
    input_size = (10, 2)
    con_window = (5, 5)
    model = Sequential()
    #model.add(LSTM(30, input_shape=(130,10), return_sequences=True, activation='relu'))
    model.add(Conv1D(300, 5, input_shape=input_size, activation='relu', padding='same' ))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(30, 5, input_shape=input_size, activation='relu', padding='same' ))


    model.add(Flatten())

    #model.add(LSTM(130, input_shape=(10,130), return_sequences=True, activation='relu'))
    #model.add(LSTM(500, activation='relu'))
    #model.add(Dense(128, input_shape=(10, 130), activation='relu' ))
    #model.add(LSTM(128, activation='relu'))
    model.add(Dense(130, activation='relu', use_bias=True))

    model.add(Dense(130, activation='relu', use_bias=True))
    model.add(Dense(130, activation='relu', use_bias=True))
    model.add(Dense(130, activation='relu', use_bias=True))
    model.add(Dense(200, activation='relu', use_bias=True))
    model.add(Dense(200, activation='relu', use_bias=True))
    model.add(Dense(200, activation='relu', use_bias=True))
    model.add(Dense(130, activation='relu', use_bias=True))
    model.add(Dense(130, activation='elu', use_bias=True))
    model.add(Dense(130, activation='elu', use_bias=True))
    model.add(Dense(130, activation='elu', use_bias=True))

    #model.add(Dropout(0.5))

    model.add(Dense(130, activation='elu', use_bias=True))
    model.add(Dense(130, activation='elu', use_bias=True))
    model.add(Dense(130, activation='elu', use_bias=True))
    model.add(Dense(130, activation='elu', use_bias=True))
    model.add(Dense(130, activation='elu', use_bias=True))
    model.add(Dense(130, activation='elu', use_bias=True))
    model.add(Dense(130, activation='elu'))

    model.add(Dense(10, activation='tanh' ))
    #model.add(Dense(9, activation='tanh', kernel_regularizer=regularizers.l1(0.01) ))
    model.add(Dense(9, activation='tanh'))
    model.summary()
    #print(max(labelsnum))
    #model.add(Activation('relu'))
    model.compile( optimizer='Adamax', 
                  loss='mean_squared_logarithmic_error', 
                  metrics=['accuracy'] )
    
    
    
    model.fit(X_tr_mstd, y_tr, epochs=250, batch_size=5,
              validation_data= [X_te_mstd, y_te])
    y_pred = model.predict(X_te_mstd)
    y_preds = np.argmax(y_pred, axis=1)
    
    #print(accuracy_score(y_preds, y_te2))
    
    """
    #X_tr_all_final = np.insert(Xtrain, 128, X_tr_means_final, axis=2)
    #X_tr_all_final = np.insert(X_tr_all_final, 129, X_tr_std_final, axis=2)
    
    X_tr_std_final = np.std(Xtrain, axis=2)
    X_tr_means_final = np.mean(Xtrain, axis=2)
    
    X_tr_mstd_final = np.expand_dims(X_tr_means_final, axis=2)
    X_tr_mstd_final = np.insert(X_tr_mstd_final, 0, X_tr_std_final, axis=2)

    
    #RANDOM FOREST, 1000, means+std random_state= 42       
    #KAGGLE DATA TRANSFORM
    X_tr_means_kaggle = np.mean(X_kaggle, axis=2)
    X_tr_std_kaggle = np.std(X_kaggle, axis=2)
    
    X_tr_mstd_kaggle = np.expand_dims(X_tr_means_kaggle, axis=2)
    X_tr_mstd_kaggle = np.insert(X_tr_mstd_kaggle, 0, X_tr_std_kaggle, axis=2)
    #X_tr_all_kaggle = np.insert(X_kaggle, 128, X_tr_means_kaggle, axis=2)
    #X_tr_all_kaggle = np.insert(X_tr_all_kaggle, 129, X_tr_std_kaggle, axis=2)
    #X_tr_mstd_kaggle = np.concatenate((X_tr_means_kaggle, X_tr_std_kaggle), axis=1)
        
    #KAGGLE SUBMIT FILE
    #model = ExtraTreesClassifier(n_estimators = 1500)
    model.fit(X_tr_mstd_final, y, epochs=247, batch_size=10)
    #model = RandomForestClassifier(n_estimators=1000)
    #model.fit(X_tr_mstd, y_tr)
    
    y_pred = model.predict(X_tr_mstd_kaggle)
    y_preds = np.argmax(y_pred, axis=1)
    labels = list(le.inverse_transform(y_preds))
    
    with open("submission.csv", "w") as fp:
        fp.write("# Id,Surface\n")
                 
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
    
    """
    
    