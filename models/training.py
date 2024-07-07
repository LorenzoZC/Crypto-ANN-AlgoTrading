import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as gnb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils.data_fetcher import fetch_and_save_data
from utils.data_preprocessor import preprocess_data

# Run CV when True, time intensive
is_cv = False

def ANN_feed_forward():
    model = Sequential([
        Input(shape=(X.shape[1],)),  # Use Input layer to specify the input shape
        Dense(units=1000, kernel_initializer='uniform', activation='relu'),
        Dense(units=500, kernel_initializer='uniform', activation='relu'),
        Dense(units=250, kernel_initializer='uniform', activation='relu'),
        Dense(units=125, kernel_initializer='uniform', activation='relu'),
        Dense(units=75, kernel_initializer='uniform', activation='relu'),
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    if not is_cv:
        ann = ANN_feed_forward()
        ann.fit(X_train, y_train, batch_size=10, epochs=50)
    
        # Predict and evaluate the ANN model
        ann_pred = ann.predict(X_test)
        ann_pred = (ann_pred >= 0.5).astype(int)  # Convert probabilities to binary outcomes
        ann_acc = accuracy_score(y_test, ann_pred)
        ann_f1 = f1_score(y_test, ann_pred)
        ann_roc_auc = roc_auc_score(y_test, ann_pred)
    
        print(f"ANN - Accuracy: {ann_acc:.4f}, F1 Score: {ann_f1:.4f}, ROC AUC: {ann_roc_auc:.4f}")
        score = ann.evaluate(X_test, y_test, batch_size = 10)
        print('\nTest loss:', score[0])
        print('Test accuracy:', score[1]) 
    
        # # Plot the ROC curve for ANN
        # fpr, tpr, _ = roc_curve(y_test, ann_pred)
        # plt.figure(figsize=(10, 6))
        # plt.plot(fpr, tpr, color='blue', lw=2, label='ANN (AUC = %0.2f)' % ann_roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic - ANN')
        # plt.legend(loc="lower right")
        # plt.show()
    else:
        ann = ANN_feed_forward()
        ann_param_grid = {
            'batch_size': [10, 20],
            'epochs': [10, 50],
            'optimizer': ['adam'],
            'units1': [500, 1000],
            'units2': [250, 500],
            'units3': [125, 250],
            'units4': [60, 125],
            'units5': [35, 75]
        }
    
        ann_random_search = RandomizedSearchCV(estimator=ann, param_distributions=ann_param_grid, n_iter=5, cv=3, verbose=2, random_state=42, n_jobs=-1)
        ann_random_search.fit(X_train, y_train, callbacks=[early_stopping])
    
        print("Best parameters found for ANN: ", ann_random_search.best_params_)
        print("Best accuracy found for ANN: ", ann_random_search.best_score_)
    
        # Evaluate the best ANN model
        best_ann_model = ann_random_search.best_estimator_
        ann_pred = best_ann_model.predict(X_test)
        ann_pred = (ann_pred >= 0.5).astype(int)  # Convert probabilities to binary outcomes
        ann_acc = accuracy_score(y_test, ann_pred)
        ann_f1 = f1_score(y_test, ann_pred)
        ann_roc_auc = roc_auc_score(y_test, ann_pred)
    
        print(f"ANN - Accuracy: {ann_acc:.4f}, F1 Score: {ann_f1:.4f}, ROC AUC: {ann_roc_auc:.4f}")
        score = ann.evaluate(X_test, y_test, batch_size = 10)
        print('\nTest loss:', score[0])
        print('Test accuracy:', score[1]) 
    
        # # Plot the ROC curve for ANN
        # fpr, tpr, _ = roc_curve(y_test, ann_pred)
        # plt.figure(figsize=(10, 6))
        # plt.plot(fpr, tpr, color='blue', lw=2, label='ANN (AUC = %0.2f)' % ann_roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic - ANN')
        # plt.legend(loc="lower right")
        # plt.show()

# reshape input to be [samples, time steps, features]
lstm_X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
lstm_X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

def create_lstm(hidden_size=32, optimizer='adam'):
    model = Sequential([
        Input(shape=(X.shape[1], 1)),  # Use Input layer to specify the input shape
        LSTM(hidden_size),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=optimizer)
    return model

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    if not is_cv:
        lstm = create_lstm()
        lstm.fit(lstm_X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stopping])
    
        # Predict and evaluate the LSTM model
        lstm_pred = lstm.predict(lstm_X_test)
        lstm_pred = (lstm_pred >= 0.5).astype(int)  # Convert probabilities to binary outcomes
        lstm_acc = accuracy_score(y_test, lstm_pred)
        lstm_f1 = f1_score(y_test, lstm_pred)
        lstm_roc_auc = roc_auc_score(y_test, lstm_pred)
    
        print(f"LSTM - Accuracy: {lstm_acc:.4f}, F1 Score: {lstm_f1:.4f}, ROC AUC: {lstm_roc_auc:.4f}")
        score = lstm.evaluate(lstm_X_test, y_test, batch_size=32)
        print('\nTest loss:', score[0])
        print('Test accuracy:', score[1])
    
        # # Plot the ROC curve for LSTM
        # fpr, tpr, _ = roc_curve(y_test, lstm_pred)
        # plt.figure(figsize=(10, 6))
        # plt.plot(fpr, tpr, color='blue', lw=2, label='LSTM (AUC = %0.2f)' % lstm_roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic - LSTM')
        # plt.legend(loc="lower right")
        # plt.show()
    else:
        lstm = create_lstm()
        lstm_param_grid = {
            'batch_size': [10, 20, 32],
            'epochs': [10, 50, 100],
            'optimizer': ['adam', 'rmsprop'],
            'hidden_size': [16, 32, 64]
        }
    
        lstm_random_search = RandomizedSearchCV(estimator=lstm, param_distributions=lstm_param_grid, n_iter=5, cv=3, verbose=2, random_state=42, n_jobs=-1)
        lstm_random_search.fit(lstm_X_train, y_train, callbacks=[early_stopping])
    
        print("Best parameters found for LSTM: ", lstm_random_search.best_params_)
        print("Best accuracy found for LSTM: ", lstm_random_search.best_score_)

        # Evaluate the best LSTM model
        best_lstm_model = lstm_random_search.best_estimator_
        lstm_pred = best_lstm_model.predict(lstm_X_test)
        lstm_pred = (lstm_pred >= 0.5).astype(int)  # Convert probabilities to binary outcomes
        lstm_acc = accuracy_score(y_test, lstm_pred)
        lstm_f1 = f1_score(y_test, lstm_pred)
        lstm_roc_auc = roc_auc_score(y_test, lstm_pred)
    
        print(f"LSTM - Accuracy: {lstm_acc:.4f}, F1 Score: {lstm_f1:.4f}, ROC AUC: {lstm_roc_auc:.4f}")
        print('\nTest loss:', score[0])
        print('Test accuracy:', score[1])
    
        # # Plot the ROC curve for LSTM
        # fpr, tpr, _ = roc_curve(y_test, lstm_pred)
        # plt.figure(figsize=(10, 6))
        # plt.plot(fpr, tpr, color='blue', lw=2, label='LSTM (AUC = %0.2f)' % lstm_roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic - LSTM')
        # plt.legend(loc="lower right")
        # plt.show()

#. Support vector Machine Model
svc = SVC(kernel = 'rbf', probability = True)  # Radial Basis Function Kernel, sigmoid: which is suitable for binary classification
svc.fit(X_train, y_train) 

svc_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)
svc_f1 = f1_score(y_test, svc_pred)
svc_roc_auc = roc_auc_score(y_test, svc.predict_proba(X_test)[:, 1])

print(f"SVC - Accuracy: {svc_acc:.4f}, F1 Score: {svc_f1:.4f}, ROC AUC: {svc_roc_auc:.4f}")

# # Plot the ROC curve for SVC
# fpr, tpr, _ = roc_curve(y_test, svc.predict_proba(X_test)[:, 1])
# plt.figure(figsize=(10, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label='SVC (AUC = %0.2f)' % svc_roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic - SVC')
# plt.legend(loc="lower right")
# plt.show()

# Gaussian Naive Bayes (GNB)
gaussian = gnb() # Gaussian Kernel, sigmoid: which is suitable for binary classification
gaussian.fit(X_train, y_train) 

gnb_pred = gaussian.predict(X_test)
gnb_acc = accuracy_score(y_test, gnb_pred)
gnb_f1 = f1_score(y_test, gnb_pred)
gnb_roc_auc = roc_auc_score(y_test, gaussian.predict_proba(X_test)[:, 1])

print(f"GNB - Accuracy: {gnb_acc:.4f}, F1 Score: {gnb_f1:.4f}, ROC AUC: {gnb_roc_auc:.4f}")

# # Plot the ROC curve for GNB
# fpr, tpr, _ = roc_curve(y_test, gaussian.predict_proba(X_test)[:, 1])
# plt.figure(figsize=(10, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label='GNB (AUC = %0.2f)' % gnb_roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic - GNB')
# plt.legend(loc="lower right")
# plt.show()

# Predict Y for each models (out of samples)
pre_lst = []

# taking values from the models run previously
ann_y_pred = ann_pred
lstm_y_pred = lstm_pred
svc_y_pred = svc_pred
gs_y_pred = gnb_pred

# Get y_prediction list for each model
pre_lst.append(('ANN', ann_y_pred))
pre_lst.append(('LSTM', lstm_y_pred))
pre_lst.append(('SVC', svc_y_pred))
pre_lst.append(('Naive Bayes', gs_y_pred))

predictions_dict = {
    'ANN': ann_y_pred,
    'LSTM': lstm_y_pred,
    'SVC': svc_y_pred,
    'GS': gs_y_pred
}

# Set threshold to categorize the y_predict
def performance_metric(threshold, y_predict):
    y_pre = [1 if x > threshold else 0 for x in y_predict]

    # 1.calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pre)
    # print("Thresholds: {}".format(thresholds),
    #      "fpr: {}".format(fpr),
    #      "tpr: {}".format(tpr))

    # 2.calculate confusion matrix
    con_mx = confusion_matrix(y_test, y_pre)
    # print("Confusion Matrix:\n{}".format(con_mx))

    # 3.calculate f1
    f1_sco = f1_score(y_test, y_pre, labels = None, pos_label = 1)
    # print("F1 Score: {}".format(f1_sco))

    # 4.calculate accuracy
    acc_score = accuracy_score(y_test, y_pre, normalize = True) 
    # print("Accuracy Score: {}".format(acc_score))

    # 5.calculate precision_score
    pre_score = precision_score(y_test, y_pre, average = 'weighted')
    # print("Precision Score: {}".format(pre_score))

    # 6.calculate recall score
    rec_score = recall_score(y_test, y_pre)
    # print("Recall Score: {}".format(rec_score))

    # 7.calculate auc score
    auc_score = auc(fpr, tpr)
    # print("AUC Score: {}".format(auc_score))

    return fpr, tpr, auc_score, f1_sco

# f1 score
name_lst = []
fpr_lst = []
tpr_lst = []
auc_score_lst = []
f1_score_lst = []

best_f1_score = 0
best_y_pred = None
best_model = None

for name, y in predictions_dict.items():
    # print('Model: %s' %name )
    fpr, tpr, auc_score, f1_sco = performance_metric(0.5, y) # set threshold = 0.5
    fpr_lst.append(fpr)
    tpr_lst.append(tpr)
    auc_score_lst.append(auc_score)
    f1_score_lst.append(f1_sco)
    name_lst.append(name)
    
    if f1_sco > best_f1_score:
        best_f1_score = f1_sco
        best_model_name = name
        best_y_pred = y # best_model_name.lower() + "_y_pred"

    # print()
    
if best_y_pred.ndim == 2 and best_y_pred.shape[1] == 1:
    best_y_pred = best_y_pred.ravel()
    
print(f"The best model is {best_model_name} with an F1 score of {best_f1_score}")
print(f"The corresponding y_pred variable name is '{best_model_name.lower()}_y_pred'")

data['best_y_pred'] = np.NaN
data.iloc[(len(data) - len(best_y_pred)):, -1:] = best_y_pred

"""
from backtrader PandasDirectData:
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', 6),
"""

df = pd.DataFrame(index = data['OpenTime']) 
df['Open'] = data['Open'].values      
df['High'] = data['High'].values      
df['Low'] = data['Low'].values
df['Close'] = data['Close'].values
df['Volume'] = data['Volume'].values
df['openinterest'] = data['best_y_pred'].values

# Clean

clean_df = df[np.isfinite(df['openinterest'])]
clean_df.index = pd.to_datetime(clean_df.index, unit='s')
clean_df.index = clean_df.index.tz_localize('UTC')

# Ensure all other columns are numeric
for col in clean_df.columns:
    if col != 'Open':
        clean_df.loc[:, col] = pd.to_numeric(clean_df[col])
        
scaler = MinMaxScaler()
clean_df.loc[:, 'openinterest'] = scaler.fit_transform(clean_df[['openinterest']])

df = pd.DataFrame(index = data['OpenTime']) 
df['Open'] = data['Open'].values      
df['High'] = data['High'].values      
df['Low'] = data['Low'].values
df['Close'] = data['Close'].values
df['Volume'] = data['Volume'].values
df['openinterest'] = data['best_y_pred'].values

# Clean

clean_df = df[np.isfinite(df['openinterest'])]
clean_df.index = pd.to_datetime(clean_df.index, unit='s')
clean_df.index = clean_df.index.tz_localize('UTC')

# Ensure all other columns are numeric
for col in clean_df.columns:
    if col != 'Open':
        clean_df.loc[:, col] = pd.to_numeric(clean_df[col])
        
scaler = MinMaxScaler()
clean_df.loc[:, 'openinterest'] = scaler.fit_transform(clean_df[['openinterest']])
