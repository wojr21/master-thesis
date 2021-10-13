# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

os.chdir("C:/Users/Wojtek/Desktop/Uczelnia/Praca magisterska")

# %% Wczytanie zbiorów danych
x_train = pd.read_excel('x_train.xlsx', index_col = 0)
y_train = pd.read_excel('y_train.xlsx', index_col = 0).iloc[:,0]
x_validation = pd.read_excel('x_validation.xlsx', index_col = 0)
y_validation = pd.read_excel('y_validation.xlsx', index_col = 0).iloc[:,0]
x_test = pd.read_excel('x_test.xlsx', index_col = 0)
y_test = pd.read_excel('y_test.xlsx', index_col = 0).iloc[:,0]

# %% Wczytanie biblioteki potrzebnej do standaryzacji
from sklearn.preprocessing import StandardScaler

# %% Standaryzacja zmiennych objasnianych w zbiorze treningowym
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled, columns = x_train.columns)

# %% Standaryzacja zmiennych objasnianych w zbiorze walidacyjnym
x_validation_scaled = (x_validation - np.mean(x_train)) / np.std(x_train)

# %% Standaryzacja zbioru testowego na podstawie statystyk zbioru treningowego
x_test_scaled = (x_test - np.mean(x_train)) / np.std(x_train)

# %% Zdefiniowanie funkcji do znalezienia najlepszego modelu sieci neuronowych
def NN_model(X_train_data, y_train_data,
             X_validation_data, y_validation_data,
             X_test_data, y_test_data):
    global y_prob_NN, y_pred_NN, best_model_NN, cm_NN, classifier_NN
    from datetime import datetime
    import tensorflow as tf
    print(tf.__version__)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    a = int(0)
    progress = int(0)
    current_time = datetime.now().strftime("%H:%M:%S")
    np.random.seed(123)
    tf.random.set_seed(123)
    for opt_fun in ['RMSprop', 'adam', 'SGD']:
        for loss_fun in ['binary_crossentropy', 'mean_squared_error']:
            for act_fun in ['relu', 'tanh']:
                for num_units in range(10, 56, 5):
                    model = Sequential()
                    model.add(Dense(units = num_units,
                                    activation = act_fun,
                                    input_shape = (X_train_data.shape[1],)))
                    model.add(Dense(units = int(num_units / 3),
                                    activation = act_fun))
                    model.add(Dense(units = 1,
                                    activation = 'sigmoid'))
                    model.compile(optimizer = opt_fun,
                                  loss = loss_fun,
                                  metrics = ['accuracy', 'Recall', 'Precision'])
                    history = model.fit(X_train_data,
                                        y_train_data,
                                        epochs = 30,
                                        batch_size = 4,
                                        verbose = 0)
                    metrics_validation = model.evaluate(X_validation_data,
                                                        y_validation_data,
                                                        verbose = 0,
                                                        batch_size = None)[1:]
                    metrics_test = model.evaluate(X_test_data,
                                                  y_test_data,
                                                  verbose = 0,
                                                  batch_size = None)[1:]
                    acc_validation = metrics_validation[0]
                    recall_validation = metrics_validation[1]
                    precision_validation = metrics_validation[2]
                    acc_test = metrics_test[0]
                    recall_test = metrics_test[1]
                    if abs((history.history["accuracy"][-1] - acc_validation)) < 0.03:
                        if (2 * (precision_validation * recall_validation) \
                            / (precision_validation + recall_validation)) > a:
                            a = 2 * (precision_validation * recall_validation) \
                                / (precision_validation + recall_validation)
                            classifier_NN = model
                            best_model_NN = pd.DataFrame({'Training Data Accuracy': history.history["accuracy"][-1],
                                                          'Validation Data Accuracy': [acc_validation],
                                                          'Test Data Accuracy': [acc_test],
                                                          'Validation Data Recall': [recall_validation],
                                                          'Test Data Recall': [recall_test],
                                                          'Units': [num_units],
                                                          'Activation function': [act_fun],
                                                          'Loss Function': [loss_fun],
                                                          'Optimizer': [opt_fun]})
                            y_prob_NN = pd.DataFrame(model.predict(X_test_data))
                            y_pred_NN = pd.DataFrame(model.predict_classes(X_test_data))
                            df_history = pd.DataFrame(history.history).iloc[:,:2]
                    progress += 1
                    if progress % 6 == 0:
                        print(str(int(progress * 100 / 120)) + "%   " + str(progress) + "/120")
    print(20 * '-')           
    print("Start time =", current_time)
    current_time = datetime.now().strftime("%H:%M:%S")
    print("End time =", current_time)
    print(20 * '-')   
    cm_NN = pd.DataFrame(confusion_matrix(y_test_data, y_pred_NN),
                      index = ['true_0', 'true_1'],
                      columns = ['pred_0', 'pred_1'])
    print(best_model_NN)
    print(20 * '-')   
    print(cm_NN)
    print(20 * '-')  
    print(classification_report(y_test_data, y_pred_NN))
    
    fig, ax = plt.subplots()
    ax.plot(df_history["accuracy"], label = "Accuracy")
    ax.plot(df_history["loss"], label = "Loss")
    ax.set_xlabel("Iterations")
    ax.set_title("Training NN Model", size = 12)
    ax.legend(loc="right")
    plt.savefig("Training NN Model.pdf")
    plt.show()

# %% Znalezienie najlepszego modelu sieci neuronowych
NN_model(x_train_scaled, y_train,
         x_validation_scaled, y_validation,
         x_test_scaled, y_test)

# %% Zdefiniowanie funkcji do znalezienia najlepszego modelu regresji logistycznej
def LR_model(X_train_data, y_train_data,
             X_validation_data, y_validation_data,
             X_test_data, y_test_data):
    global y_prob_LR, y_pred_LR, best_model_LR, cm_LR, beta_LR, classifier_LR
    from datetime import datetime
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    a = int(0)
    progress = int(0)
    beta_LR = pd.DataFrame({'Features': ['Intercept'] + list(X_train_data.columns)})
    current_time = datetime.now().strftime("%H:%M:%S")
    np.random.seed(123)
    for solver in ['liblinear', 'lbfgs', 'newton-cg']:
        classifier = LogisticRegression(solver = solver)
        classifier.fit(X_train_data, y_train_data)
        acc_train = classifier.score(X_train_data, y_train_data)
        acc_validation = classifier.score(X_validation_data, y_validation_data)
        acc_test = classifier.score(X_test_data, y_test_data)
        pred_validation = classifier.predict(X_validation_data)
        pred = classifier.predict(X_test_data)
        recall_validation = recall_score(y_validation_data, pred_validation)
        recall_test = recall_score(y_test_data, pred)
        f1 = f1_score(y_validation_data, pred_validation)
        if abs(acc_train - acc_validation) < 0.05:
            if f1 > a:
                a = f1
                classifier_LR = classifier
                best_model_LR = pd.DataFrame({'Training Data Accuracy': [acc_train],
                                              'Validation Data Accuracy': [acc_validation],
                                              'Test Data Accuracy': [acc_test],
                                              'Validation Data Recall': [recall_validation],
                                              'Test Data Recall': [recall_test],
                                              'Solver': [solver]})
                y_prob_LR = pd.DataFrame(classifier.predict_proba(X_test_data)[:,1])
                y_pred_LR = pd.DataFrame(pred)
                beta_LR['Beta_value'] = np.vstack((classifier.intercept_,
                                                   classifier.coef_.reshape((-1, 1))))
        progress += 1
        if (progress * 100 / 3) % 2 == 0:
            print(str(int(progress * 100 / 3)) + "%   " + str(progress) + "/3")
    print(20 * '-')           
    print("Start time =", current_time)
    current_time = datetime.now().strftime("%H:%M:%S")
    print("End time =", current_time)
    print(20 * '-')
    cm_LR = pd.DataFrame(confusion_matrix(y_test_data, y_pred_LR),
                         index = ['true_0', 'true_1'],
                         columns = ['pred_0', 'pred_1'])
    print(best_model_LR)
    print(20 * '-')   
    print(cm_LR)
    print(20 * '-')  
    print(classification_report(y_test_data, y_pred_LR))
    print(20 * '-') 
    print(beta_LR)

# %% Znalezienie najlepszego modelu regresji logistycznej
LR_model(x_train_scaled, y_train,
         x_validation_scaled, y_validation,
         x_test_scaled, y_test)

# %% Zdefiniowanie funkcji do znalezienia najlepszego modelu drzew decyzyjnych
def DT_model(X_train_data, y_train_data,
             X_validation_data, y_validation_data,
             X_test_data, y_test_data):
    global y_prob_DT, y_pred_DT, best_model_DT, cm_DT, importance_DT, classifier_DT
    from datetime import datetime
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    a = int(0)
    progress = int(0)
    importance_DT = pd.DataFrame({'Features': X_train_data.columns})
    current_time = datetime.now().strftime("%H:%M:%S")
    np.random.seed(123)
    for max_dep in range(2, 11):
        for min_sam_leaf in range(30, 121, 2):
            classifier = DecisionTreeClassifier(max_depth = max_dep,
                                                min_samples_leaf = min_sam_leaf)
            classifier.fit(X_train_data, y_train_data)
            acc_train = classifier.score(X_train_data, y_train_data)
            acc_validation = classifier.score(X_validation_data, y_validation_data)
            acc_test = classifier.score(X_test_data, y_test_data)
            pred_validation = classifier.predict(X_validation_data)
            pred = classifier.predict(X_test_data)
            recall_validation = recall_score(y_validation_data, pred_validation)
            recall_test = recall_score(y_test_data, pred)
            f1 = f1_score(y_validation_data, pred_validation)
            if abs(acc_train - acc_validation) < 0.03:
                if f1 > a:
                    a = f1
                    classifier_DT = classifier
                    best_model_DT = pd.DataFrame({'Training Data Accuracy': [acc_train],
                                                  'Validation Data Accuracy': [acc_validation],
                                                  'Test Data Accuracy': [acc_test],
                                                  'Validation Data Recall': [recall_validation],
                                                  'Test Data Recall': [recall_test],
                                                  'Max Depth': [max_dep],
                                                  'Min Samples Leaf': [min_sam_leaf]})
                    y_prob_DT = pd.DataFrame(classifier.predict_proba(X_test_data)[:,1])
                    y_pred_DT = pd.DataFrame(pred)
                    importance_DT['Importance'] = classifier.feature_importances_
            progress += 1
            if progress % 43 == 0:
                print(str(int(progress * 100 / 387)) + "%   " + str(progress) + "/387")
    print(20 * '-')           
    print("Start time =", current_time)
    current_time = datetime.now().strftime("%H:%M:%S")
    print("End time =", current_time)
    print(20 * '-')
    cm_DT = pd.DataFrame(confusion_matrix(y_test_data, y_pred_DT),
                         index = ['true_0', 'true_1'],
                         columns = ['pred_0', 'pred_1'])
    print(best_model_DT)
    print(20 * '-')   
    print(cm_DT)
    print(20 * '-')  
    print(classification_report(y_test_data, y_pred_DT))
    print(20 * '-')
    print(importance_DT)

# %% Znalezienie najlepszego modelu drzew decyzyjnych
DT_model(x_train_scaled, y_train,
         x_validation_scaled, y_validation,
         x_test_scaled, y_test)

# %% Zdefiniowanie funkcji do znalezienia najlepszego modelu lasu losowego
def RF_model(X_train_data, y_train_data,
             X_validation_data, y_validation_data,
             X_test_data, y_test_data):
    global y_prob_RF, y_pred_RF, best_model_RF, cm_RF, importance_RF, classifier_RF
    from datetime import datetime
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    a = int(0)
    progress = int(0)
    importance_RF = pd.DataFrame({'Features': X_train_data.columns})
    current_time = datetime.now().strftime("%H:%M:%S")
    np.random.seed(123)
    for n_trees in range(60, 211, 30):
        for max_dep in range(1, 11):
            for min_sam_leaf in range(30, 71, 2):
                classifier = RandomForestClassifier(n_estimators = n_trees,
                                                    max_depth = max_dep,
                                                    min_samples_leaf = min_sam_leaf)
                classifier.fit(X_train_data, y_train_data)
                acc_train = classifier.score(X_train_data, y_train_data)
                acc_validation = classifier.score(X_validation_data, y_validation_data)
                acc_test = classifier.score(X_test_data, y_test_data)
                pred_validation = classifier.predict(X_validation_data)
                pred = classifier.predict(X_test_data)
                recall_validation = recall_score(y_validation_data, pred_validation)
                recall_test = recall_score(y_test_data, pred)
                f1 = f1_score(y_validation_data, pred_validation)
                if abs(acc_train - acc_validation) < 0.03:
                    if f1 > a:
                        a = f1
                        classifier_RF = classifier
                        best_model_RF = pd.DataFrame({'Training Data Accuracy': [acc_train],
                                                      'Validation Data Accuracy': [acc_validation],
                                                      'Test Data Accuracy': [acc_test],
                                                      'Validation Data Recall': [recall_validation],
                                                      'Test Data Recall': [recall_test],
                                                      'Number of Trees': [n_trees],
                                                      'Max Depth': [max_dep],
                                                      'Min Samples Leaf': [min_sam_leaf]})
                        y_prob_RF = pd.DataFrame(classifier.predict_proba(X_test_data)[:,1])
                        y_pred_RF = pd.DataFrame(pred)
                        importance_RF['Importance'] = classifier.feature_importances_
                progress += 1
                if progress % 60 == 0:
                    print(str(int(progress * 100 / 1260)) + "%   " + str(progress) + "/1260")
    print(20 * '-')           
    print("Start time =", current_time)
    current_time = datetime.now().strftime("%H:%M:%S")
    print("End time =", current_time)
    print(20 * '-')
    cm_RF = pd.DataFrame(confusion_matrix(y_test_data, y_pred_RF),
                         index = ['true_0', 'true_1'],
                         columns = ['pred_0', 'pred_1'])
    print(best_model_RF)
    print(20 * '-')   
    print(cm_RF)
    print(20 * '-')  
    print(classification_report(y_test_data, y_pred_RF))
    print(20 * '-')
    print(importance_RF)

# %% Znalezienie najlepszego modelu lasu losowego
RF_model(x_train_scaled, y_train,
         x_validation_scaled, y_validation,
         x_test_scaled, y_test)

# %% Zdefiniowanie funkcji do znalezienia najlepszego modelu SVM
def SVM_model(X_train_data, y_train_data,
              X_validation_data, y_validation_data,
              X_test_data, y_test_data):
    global y_prob_SVM, y_pred_SVM, best_model_SVM, cm_SVM, classifier_SVM
    from datetime import datetime
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    a = int(0)
    progress = int(0)
    current_time = datetime.now().strftime("%H:%M:%S")
    np.random.seed(123)
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for gamma in list(np.arange(0.0005, 0.2005, 0.0005)):
            classifier = SVC(kernel = kernel, gamma = gamma, probability = True)
            classifier.fit(X_train_data, y_train_data)
            acc_train = classifier.score(X_train_data, y_train_data)
            acc_validation = classifier.score(X_validation_data, y_validation_data)
            acc_test = classifier.score(X_test_data, y_test_data)
            pred_validation = classifier.predict(X_validation_data)
            pred = classifier.predict(X_test_data)
            recall_validation = recall_score(y_validation_data, pred_validation)
            recall_test = recall_score(y_test_data, pred)
            f1 = f1_score(y_validation_data, pred_validation)
            if abs(acc_train - acc_validation) < 0.03:
                if f1 > a:
                    a = f1
                    classifier_SVM = classifier
                    best_model_SVM = pd.DataFrame({'Training Data Accuracy': [acc_train],
                                                   'Validation Data Accuracy': [acc_validation],
                                                   'Test Data Accuracy': [acc_test],
                                                   'Validation Data Recall': [recall_validation],
                                                   'Test Data Recall': [recall_test],
                                                   'Kernel': [kernel],
                                                   'Gamma': [gamma]})
                    y_prob_SVM = pd.DataFrame(classifier.predict_proba(X_test_data)[:,1])
                    y_pred_SVM = pd.DataFrame(pred)
            progress += 1
            if progress % 80 == 0:
                print(str(int(progress * 100 / 1600)) + "%   " + str(progress) + "/1600")
    print(20 * '-')           
    print("Start time =", current_time)
    current_time = datetime.now().strftime("%H:%M:%S")
    print("End time =", current_time)
    print(20 * '-')
    cm_SVM = pd.DataFrame(confusion_matrix(y_test_data, y_pred_SVM),
                          index = ['true_0', 'true_1'],
                          columns = ['pred_0', 'pred_1'])
    print(best_model_SVM)
    print(20 * '-')   
    print(cm_SVM)
    print(20 * '-')  
    print(classification_report(y_test_data, y_pred_SVM))
    print(20 * '-')

# %% Znalezienie najlepszego modelu SVM
SVM_model(x_train_scaled, y_train,
          x_validation_scaled, y_validation,
          x_test_scaled, y_test)

# %% Zdefiniowanie funkcji do znalezienia najlepszego modelu KNN
def KNN_model(X_train_data, y_train_data,
              X_validation_data, y_validation_data,
              X_test_data, y_test_data):
    global y_prob_KNN, y_pred_KNN, best_model_KNN, cm_KNN, importance_KNN, classifier_KNN
    from datetime import datetime
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.inspection import permutation_importance
    a = int(0)
    progress = int(0)
    importance_KNN = pd.DataFrame({'Features': X_train_data.columns})
    current_time = datetime.now().strftime("%H:%M:%S")
    np.random.seed(123)
    for nn in range(5, 26):
        for wgts in (['uniform', 'distance']):
            for alg in (['ball_tree', 'kd_tree', 'brute']):
                for leaf in range(10, 51, 5):
                    for p in [1, 2]:
                        classifier = KNeighborsClassifier(n_neighbors = nn,
                                                          weights = wgts,
                                                          algorithm = alg,
                                                          leaf_size = leaf,
                                                          p = p)
                        classifier.fit(X_train_data, y_train_data)
                        acc_train = classifier.score(X_train_data, y_train_data)
                        acc_validation = classifier.score(X_validation_data, y_validation_data)
                        acc_test = classifier.score(X_test_data, y_test_data)
                        pred_validation = classifier.predict(X_validation_data)
                        pred = classifier.predict(X_test_data)
                        recall_validation = recall_score(y_validation_data, pred_validation)
                        recall_test = recall_score(y_test_data, pred)
                        f1 = f1_score(y_validation_data, pred_validation)
                        if abs(acc_train - acc_validation) < 0.03:
                            if f1 > a:
                                a = f1
                                classifier_KNN = classifier
                                best_model_KNN = pd.DataFrame({'Training Data Accuracy': [acc_train],
                                                               'Validation Data Accuracy': [acc_validation],
                                                               'Test Data Accuracy': [acc_test],
                                                               'Validation Data Recall': [recall_validation],
                                                               'Test Data Recall': [recall_test],
                                                               'N_Neighbors': [nn],
                                                               'Weights': [wgts],
                                                               'Algorithm': [alg],
                                                               'Leaf Size': [leaf],
                                                               'P': [p]})
                                y_prob_KNN = pd.DataFrame(classifier.predict_proba(X_test_data)[:,1])
                                y_pred_KNN = pd.DataFrame(pred)
                                importance_KNN['Importance'] = permutation_importance(classifier,
                                                                                      X_train_data,
                                                                                      y_train_data,
                                                                                      scoring = 'accuracy').importances_mean
                        progress += 1
                        if progress % 63 == 0:
                            print(str(int(progress * 100 / 2268)) + "%   " + str(progress) + "/2268")
    print(20 * '-')           
    print("Start time =", current_time)
    current_time = datetime.now().strftime("%H:%M:%S")
    print("End time =", current_time)
    print(20 * '-')
    cm_KNN = pd.DataFrame(confusion_matrix(y_test_data, y_pred_KNN),
                          index = ['true_0', 'true_1'],
                          columns = ['pred_0', 'pred_1'])
    print(best_model_KNN)
    print(20 * '-')   
    print(cm_KNN)
    print(20 * '-')  
    print(classification_report(y_test_data, y_pred_KNN))
    print(20 * '-')
    print(importance_KNN)

# %% Znalezienie najlepszego modelu KNN
KNN_model(x_train_scaled, y_train,
          x_validation_scaled, y_validation,
          x_test_scaled, y_test)

# %% Zapisanie wyników do pliku

# NN
results = pd.concat([y_test, y_pred_NN, y_prob_NN], axis = 1)
results.columns = ['True y', 'Predicted y', 'Probability']
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    results.to_excel(writer, sheet_name = 'Neural Networks',
                     startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    cm_NN.to_excel(writer, sheet_name = 'Neural Networks',
                   startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_NN.to_excel(writer, sheet_name = 'Neural Networks',
                           startrow = 1, startcol = 1)

# LR
results = pd.concat([y_test, y_pred_LR, y_prob_LR], axis = 1)
results.columns = ['True y', 'Predicted y', 'Probability']
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    results.to_excel(writer, sheet_name = 'Logistic Regression',
                     startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    cm_LR.to_excel(writer, sheet_name = 'Logistic Regression',
                   startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_LR.to_excel(writer, sheet_name = 'Logistic Regression',
                           startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    beta_LR.to_excel(writer, sheet_name = 'Logistic Regression',
                     startrow = 1, startcol = 1)

# DT
results = pd.concat([y_test, y_pred_DT, y_prob_DT], axis = 1)
results.columns = ['True y', 'Predicted y', 'Probability']
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    results.to_excel(writer, sheet_name = 'Decision Tree',
                     startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    cm_DT.to_excel(writer, sheet_name = 'Decision Tree',
                   startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_DT.to_excel(writer, sheet_name = 'Decision Tree',
                           startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    importance_DT.to_excel(writer, sheet_name = 'Decision Tree',
                           startrow = 1, startcol = 1)

# RF
results = pd.concat([y_test, y_pred_RF, y_prob_RF], axis = 1)
results.columns = ['True y', 'Predicted y', 'Probability']
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    results.to_excel(writer, sheet_name = 'Random Forest',
                     startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    cm_RF.to_excel(writer, sheet_name = 'Random Forest',
                   startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_RF.to_excel(writer, sheet_name = 'Random Forest',
                           startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    importance_RF.to_excel(writer, sheet_name = 'Random Forest',
                           startrow = 1, startcol = 1)

# SVM
results = pd.concat([y_test, y_pred_SVM, y_prob_SVM], axis = 1)
results.columns = ['True y', 'Predicted y', 'Probability']
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    results.to_excel(writer, sheet_name = 'SVM',
                     startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    cm_SVM.to_excel(writer, sheet_name = 'SVM',
                    startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_SVM.to_excel(writer, sheet_name = 'SVM',
                            startrow = 1, startcol = 1)
    
# KNN
results = pd.concat([y_test, y_pred_KNN, y_prob_KNN], axis = 1)
results.columns = ['True y', 'Predicted y', 'Probability']
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    results.to_excel(writer, sheet_name = 'KNN',
                     startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    cm_KNN.to_excel(writer, sheet_name = 'KNN',
                    startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_KNN.to_excel(writer, sheet_name = 'KNN',
                            startrow = 1, startcol = 1)
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    importance_KNN.to_excel(writer, sheet_name = 'KNN',
                            startrow = 1, startcol = 1)

# %% Wczytanie pełnego zbioru danych do sprawdzenia otrzymanych modeli
df = pd.read_excel('df_preprocessed.xlsx', index_col = 0)

# %% Sprawdzenie uzyskanych modeli przy losowaniu zbioru danych 10 razy
def check_models(df, model_NN = best_model_NN, model_LR = best_model_LR,
                 model_DT = best_model_DT, model_RF = best_model_RF,
                 model_SVM = best_model_SVM, model_KNN = best_model_KNN,
                 NN = classifier_NN, LR = classifier_LR, DT = classifier_DT,
                 RF = classifier_RF, SVM = classifier_SVM, KNN = classifier_KNN):
    global best_model_NN, best_model_LR, best_model_DT, best_model_RF, best_model_SVM, best_model_KNN
    from sklearn.metrics import recall_score
    progress = int(0)
    for j in range(1, 11):
        X0 = df[df["upadlosc"] == 0].reset_index(drop = True)
        X1 = df[df["upadlosc"] == 1].reset_index(drop = True)
        X0 = X0.sample(int(9 * len(X1)), axis = 0).reset_index(drop = True)
        X0_sample = X0.sample(int(0.2 * len(X0)), axis = 0)
        X1_sample = X1.sample(int(0.2 * len(X1)), axis = 0)
        X_test = pd.concat([X0_sample, X1_sample], axis = 0).sample(frac = 1).reset_index(drop = True)
        y_test = X_test.pop('upadlosc')
        
        X_test_scaled =  (X_test - np.mean(x_train)) / np.std(x_train)
        
        # NN model check
        metrics_test = NN.evaluate(X_test_scaled,
                                   y_test,
                                   verbose = 0,
                                   batch_size = None)[1:]
        acc_test_NN = metrics_test[0]
        recall_test_NN = metrics_test[1]
        NN_model_1 = pd.DataFrame({'Training Data Accuracy': ['-'],
                                   'Validation Data Accuracy': ['-'],
                                   'Test Data Accuracy': [acc_test_NN],
                                   'Validation Data Recall': ['-'],
                                   'Test Data Recall': [recall_test_NN],
                                   'Units': [model_NN.iloc[0, -4]],
                                   'Activation function': [model_NN.iloc[0, -3]],
                                   'Loss Function': [model_NN.iloc[0, -2]],
                                   'Optimizer': [model_NN.iloc[0, -1]]})
        best_model_NN = best_model_NN.append(NN_model_1, ignore_index = True)
        
        # LR model check
        acc_test_LR = LR.score(X_test_scaled, y_test)
        pred_LR = LR.predict(X_test_scaled)
        recall_test_LR = recall_score(y_test, pred_LR)
        LR_model_1 = pd.DataFrame({'Training Data Accuracy': ['-'],
                                   'Validation Data Accuracy': ['-'],
                                   'Test Data Accuracy': [acc_test_LR],
                                   'Validation Data Recall': ['-'],
                                   'Test Data Recall': [recall_test_LR],
                                   'Solver': [model_LR.iloc[0, -1]]})
        best_model_LR = best_model_LR.append(LR_model_1, ignore_index = True)
        
        # DT model check
        acc_test_DT = DT.score(X_test_scaled, y_test)
        pred_DT = DT.predict(X_test_scaled)
        recall_test_DT = recall_score(y_test, pred_DT) 
        DT_model_1 = pd.DataFrame({'Training Data Accuracy': ['-'],
                                   'Validation Data Accuracy': ['-'],
                                   'Test Data Accuracy': [acc_test_DT],
                                   'Validation Data Recall': ['-'],
                                   'Test Data Recall': [recall_test_DT],
                                   'Max Depth': [model_DT.iloc[0, -2]],
                                   'Min Samples Leaf': [model_DT.iloc[0, -1]]})
        best_model_DT = best_model_DT.append(DT_model_1, ignore_index = True)
        
        # RF model check
        acc_test_RF = RF.score(X_test_scaled, y_test)
        pred_RF = RF.predict(X_test_scaled)
        recall_test_RF = recall_score(y_test, pred_RF)
        RF_model_1 = pd.DataFrame({'Training Data Accuracy': ['-'],
                                   'Validation Data Accuracy': ['-'],
                                   'Test Data Accuracy': [acc_test_RF],
                                   'Validation Data Recall': ['-'],
                                   'Test Data Recall': [recall_test_RF],
                                   'Number of Trees': [model_RF.iloc[0, -3]],
                                   'Max Depth': [model_RF.iloc[0, -2]], 
                                   'Min Samples Leaf': [model_RF.iloc[0, -1]]})
        best_model_RF = best_model_RF.append(RF_model_1, ignore_index = True)
        
        # SVM model check  
        acc_test_SVM = SVM.score(X_test_scaled, y_test)
        pred_SVM = SVM.predict(X_test_scaled)
        recall_test_SVM = recall_score(y_test, pred_SVM)
        SVM_model_1 = pd.DataFrame({'Training Data Accuracy': ['-'],
                                    'Validation Data Accuracy': ['-'],
                                    'Test Data Accuracy': [acc_test_SVM],
                                    'Validation Data Recall': ['-'],
                                    'Test Data Recall': [recall_test_SVM],
                                    'Kernel': [model_SVM.iloc[0, -2]],
                                    'Gamma': [model_SVM.iloc[0, -1]]})
        best_model_SVM = best_model_SVM.append(SVM_model_1, ignore_index = True)
        
        # KNN model check
        acc_test_KNN = KNN.score(X_test_scaled, y_test)
        pred_KNN = KNN.predict(X_test_scaled)
        recall_test_KNN = recall_score(y_test, pred_KNN)
        KNN_model_1 = pd.DataFrame({'Training Data Accuracy': ['-'],
                                    'Validation Data Accuracy': ['-'],
                                    'Test Data Accuracy': [acc_test_KNN],
                                    'Validation Data Recall': ['-'],
                                    'Test Data Recall': [recall_test_KNN],
                                    'N_Neighbors': [model_KNN.iloc[0, -5]],
                                    'Weights': [model_KNN.iloc[0, -4]],
                                    'Algorithm': [model_KNN.iloc[0, -3]],
                                    'Leaf Size': [model_KNN.iloc[0, -2]],
                                    'P': [model_KNN.iloc[0, -1]]})
        best_model_KNN = best_model_KNN.append(KNN_model_1, ignore_index = True)
        
        progress += 1
        print(str(int(progress * 100 / 10)) + "%   " + str(progress) + "/10")
    
    print(best_model_NN)
    print(20 * '-')
    print(best_model_LR)
    print(20 * '-')
    print(best_model_DT)
    print(20 * '-')
    print(best_model_RF)
    print(20 * '-')
    print(best_model_SVM)
    print(20 * '-')
    print(best_model_KNN)

# %% Sprawdzenie modeli
check_models(df)

# %% Wykresy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fig, ax = plt.subplots()
ax.plot(best_model_NN["Test Data Accuracy"], label = "Test Data Accuracy")
ax.plot(best_model_NN["Test Data Recall"], label = "Test Data Recall")
ax.set_xlabel("Dataset")
ax.set_title("Model NN Check", size = 12)
ax.set_ylim((0, 1.1))
ax.legend(loc="lower left")
plt.savefig("Model NN Check.pdf")

fig, ax = plt.subplots()
ax.plot(best_model_LR["Test Data Accuracy"], label = "Test Data Accuracy")
ax.plot(best_model_LR["Test Data Recall"], label = "Test Data Recall")
ax.set_xlabel("Dataset")
ax.set_title("Model LR Check", size = 12)
ax.set_ylim((0, 1.1))
ax.legend(loc="lower left")
plt.savefig("Model LR Check.pdf")

fig, ax = plt.subplots()
ax.plot(best_model_DT["Test Data Accuracy"], label = "Test Data Accuracy")
ax.plot(best_model_DT["Test Data Recall"], label = "Test Data Recall")
ax.set_xlabel("Dataset")
ax.set_title("Model DT Check", size = 12)
ax.set_ylim((0, 1.1))
ax.legend(loc="lower left")
plt.savefig("Model DT Check.pdf")

fig, ax = plt.subplots()
ax.plot(best_model_RF["Test Data Accuracy"], label = "Test Data Accuracy")
ax.plot(best_model_RF["Test Data Recall"], label = "Test Data Recall")
ax.set_xlabel("Dataset")
ax.set_title("Model RF Check", size = 12)
ax.set_ylim((0, 1.1))
ax.legend(loc="lower left")
plt.savefig("Model RF Check.pdf")

fig, ax = plt.subplots()
ax.plot(best_model_SVM["Test Data Accuracy"], label = "Test Data Accuracy")
ax.plot(best_model_SVM["Test Data Recall"], label = "Test Data Recall")
ax.set_xlabel("Dataset")
ax.set_title("Model SVM Check", size = 12)
ax.set_ylim((0, 1.1))
ax.legend(loc="lower left")
plt.savefig("Model SVM Check.pdf")

fig, ax = plt.subplots()
ax.plot(best_model_KNN["Test Data Accuracy"], label = "Test Data Accuracy")
ax.plot(best_model_KNN["Test Data Recall"], label = "Test Data Recall")
ax.set_xlabel("Dataset")
ax.set_title("Model KNN Check", size = 12)
ax.set_ylim((0, 1.1))
ax.legend(loc="lower left")
plt.savefig("Model KNN Check.pdf")

# %% Wyniki dla uzyskanych modeli przy 10-krotnym losowaniu zbiorów danych

# NN
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_NN.to_excel(writer, sheet_name = 'NN check',
                           startrow = 1, startcol = 1)

# LR
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_LR.to_excel(writer, sheet_name = 'LR check',
                           startrow = 1, startcol = 1)

# DT
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_DT.to_excel(writer, sheet_name = 'DT check',
                           startrow = 1, startcol = 1)

# RF
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_RF.to_excel(writer, sheet_name = 'RF check',
                           startrow = 1, startcol = 1)

# SVM
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_SVM.to_excel(writer, sheet_name = 'SVM check',
                            startrow = 1, startcol = 1)
    
# KNN
with pd.ExcelWriter('Wyniki.xlsx', mode = 'a') as writer:
    best_model_KNN.to_excel(writer, sheet_name = 'KNN check',
                            startrow = 1, startcol = 1)