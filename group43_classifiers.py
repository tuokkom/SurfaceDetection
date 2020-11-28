
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
#from xgboost import XGBClassifier


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
    
    #split the data to training and testing
    shuffler = GroupShuffleSplit(test_size=0.20, random_state=42)
   
    for train, test in shuffler.split(Xtrain, labelsnum, groups):
        #print(train)
        #print(test)
        X_tr = Xtrain[train]
        y_tr = labelsnum[train]
        
        X_te = Xtrain[test]
        y_te = labelsnum[test]


    #RESIZE METHOD

    X_tr_resize = np.resize(X_tr, (X_tr.shape[0], 1280))
    X_te_resize = np.resize(X_te, (X_te.shape[0], 1280))

    
    classifiers = [LinearDiscriminantAnalysis(), 
                   SVC(kernel='rbf', C=0.20),
                   SVC(kernel='rbf', C=0.50), 
                   SVC(kernel='rbf', C=1), 
                   SVC(kernel='rbf', C=2.5),
                   LogisticRegression(penalty='l1'),
                   LogisticRegression(penalty='l2'),
                   RandomForestClassifier(n_estimators=1000),
                   ExtraTreesClassifier(n_estimators=1000)]
    
    clf_names = ['LDA', 
                 'SVC kernel=0,2', 
                 'SVC kernel=0,5',
                 'SVC kernel=1', 
                 'SVC kernel=1.5',
                 'Logistic Regression l1', 
                 'Logistic Regression l2',
                 'Random Forest Classifier', 
                 'Extremely random forest',
                 'XGBClassifier gbtree']


    resize_scores = []
    #resize_scorescv = []
    all_preds = []
    tarkeys1=[]

#    print('RESIZE METHOD')
#    for clf, name in zip(classifiers, clf_names):
#        #cv_scores = cross_val_score(clf, X_for_cv, labelsnum, cv=groups_ptr )
#        clf.fit(X_tr_resize, y_tr)
#        predictions = clf.predict(X_te_resize)
#        #all_preds.append(predictions)
#        resize_scores.append(accuracy_score(predictions, y_te))
#        
#        if name in ['Random Forest Classifier', 
#                 'Extremely random forest']:
#            tarkeys1.append(clf.predict_proba(X_te_resize))
#            print('COEFFFFF LUE ASDASD \n ------------------ \n', clf.predict_proba(X_te_resize))
#        if accuracy_score(predictions, y_te) > 0.5:
#            all_preds.append(predictions)
        
        #resize_scorescv.append(cv_scores)
        #cv_scores = []
        
    #for clf_cv in classifiers:
        #cv_scores = cross_val_score(clf_cv, X_for_cv, labelsnum, cv=groups_ptr)
        #resize_scorescv.append(cv_scores)
#        if name == 'XGB Classifier':
#            pred = [round(value) for value in predictions]
#            print('Name: ', name, 'scores: ', accuracy_score(pred, y_te))
#        else:
#            print('Name: ', name, 'scores: ', accuracy_score(predictions, y_te))




    #MEAN METHOD

    print('MEAN METHOD')
    X_tr_means = np.mean(X_tr, axis=2)
    X_te_means = np.mean(X_te, axis=2)

    mean_scores = []
    for clf_m, name in zip(classifiers, clf_names):
        clf_m.fit(X_tr_means, y_tr)
        predictions_m = clf_m.predict(X_te_means)
        all_preds.append(predictions_m)
        
        mean_scores.append(accuracy_score(predictions_m, y_te))
        
        
        if name in ['Random Forest Classifier', 
                 'Extremely random forest', 'AdaBoostClassifier']:
            tarkeys1.append(clf_m.predict_proba(X_te_means))




    #MEAN + STD METHOD
    X_tr_std = np.std(X_tr, axis=2)
    X_te_std = np.std(X_te, axis=2)

    X_tr_mstd = np.concatenate((X_tr_means, X_tr_std), axis=1)
    X_te_mstd = np.concatenate((X_te_means, X_te_std), axis=1)

    std_scores = []
    print('STD METHOD')
    for clf_ms, name in zip(classifiers, clf_names):
        clf_ms.fit(X_tr_mstd, y_tr)
        predictions_ms = clf_ms.predict(X_te_mstd)
        
        all_preds.append(predictions_ms)
        
        std_scores.append(accuracy_score(predictions_ms, y_te))
        
        
        if name in ['Random Forest Classifier', 
                 'Extremely random forest', 'AdaBoostClassifier']:
            tarkeys1.append(clf_ms.predict_proba(X_te_mstd))

        
    X_tr_all = np.insert(X_tr, 128, X_tr_means, axis=2)
    X_te_all = np.insert(X_te, 128, X_te_means, axis=2) 
    
    X_tr_all = np.insert(X_tr_all, 129, X_tr_std, axis=2)
    X_te_all = np.insert(X_te_all, 129, X_te_std, axis=2)
    
    X_tr_all = np.resize(X_tr_all, (X_tr_all.shape[0], 1300))
    X_te_all = np.resize(X_te_all, (X_te_all.shape[0], 1300))
    
    all_scores = []
    print('KAIKKI METHOD')
    for clf_all, name in zip(classifiers, clf_names):
        clf_all.fit(X_tr_all, y_tr)
        predictions_all = clf_all.predict(X_te_all)
        all_scores.append(accuracy_score(predictions_all, y_te))
        
        if name in ['Random Forest Classifier', 
                 'Extremely random forest', 'AdaBoostClassifier']:
            tarkeys1.append(clf_all.predict_proba(X_te_all))
            #print('COEFFFFF LUE ASDASD \n ------------------ \n', clf_all.predict_proba(X_te_all))
        
        
        
        
#    print('RESIZE SCORES \n ------------------')
#    for score, name in zip(resize_scores, clf_names):
#        print(name, 'Score with resize', score)
        
        
    print('MEAN SCORES \n ------------------')
    for score, name in zip(mean_scores, clf_names):
        print(name, 'Score with means', score)
        
    print('STD + MEAN SCORES \n ------------------')
    for score, name in zip(std_scores, clf_names):
        print(name, 'Score with std + means', score)
    
    """
    all_preds = np.asarray(all_preds)
    most_common = []
    for ind in range(0, len(all_preds[0])):
        guess = np.bincount(all_preds[:, ind])
        class_num = np.argmax(guess)
        most_common.append(class_num)
    """
    
    #Best guess obtained from tree classifiers predict_proba method
    proba_sum = np.sum(tarkeys1, axis=0)
    most_common = np.argmax(proba_sum, axis=1)
    
    print('Best guess', accuracy_score(most_common, y_te))
        
    print('ALL DATA + MEANS + STD SCORES \n ------------------')
    for score, name in zip(all_scores, clf_names):
        print(name, 'Score with all data + std + means', score)
    
    
    #Creating submission file
    
    """
    X_tr_std_final = np.std(Xtrain, axis=2)
    X_tr_means_final = np.mean(Xtrain, axis=2)

    X_tr_mstd_final = np.concatenate((X_tr_means_final, X_tr_std_final), axis=1)
    
    
    X_tr_all_final = np.insert(Xtrain, 128, X_tr_means_final, axis=2)
    
    X_tr_all_final = np.insert(X_tr_all_final, 129, X_tr_std_final, axis=2)
    X_tr_all_final = np.resize(X_tr_all_final, (X_tr_all_final.shape[0], 1300))
    
    
    #Extremely RANDOM FOREST, 3000, means+std       
    #KAGGLE DATA TRANSFORM
    X_tr_means_kaggle = np.mean(X_kaggle, axis=2)
    X_tr_std_kaggle = np.std(X_kaggle, axis=2)
    
    X_tr_all_kaggle = np.insert(X_kaggle, 128, X_tr_means_kaggle, axis=2)
    X_tr_all_kaggle = np.insert(X_tr_all_kaggle, 129, X_tr_std_kaggle, axis=2)
    X_tr_all_kaggle = np.resize(X_tr_all_kaggle, (X_tr_all_kaggle.shape[0], 1300))
    #X_tr_mstd_kaggle = np.concatenate((X_tr_means_kaggle, X_tr_std_kaggle), axis=1)
        
    #KAGGLE SUBMIT FILE
    model = ExtraTreesClassifier(n_estimators = 3000)
    model.fit(X_tr_mstd_final, labelsnum)
    
    y_pred = model.predict(X_tr_all_kaggle)
    labels = list(le.inverse_transform(y_pred))
    
    with open("submission.csv", "w") as fp:
        fp.write("# Id,Surface\n")
                 
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
    """
