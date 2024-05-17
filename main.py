#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import joblib
from tqdm import tqdm
import neptune
import os
import logging
from datetime import datetime
from feature_lists import *
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, Bidirectional, GRU, LSTM, Dropout, concatenate, Conv1D
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# %%
logging.basicConfig(level=logging.INFO)

df = pd.read_csv('Cardio_healthy_sport_master_file.csv',decimal=',')
df = df[['ID badania','Grupa','Płeć [K/M]','Wiek','Masa','Wzrost','BMI']]
df = df.rename(columns={'ID badania':'ID','Płeć [K/M]':'Sex','Wiek':'Age','Masa':'Weight','Wzrost':'Height'})
df['Sex'] = [-1 if sex=='M' else 1 for sex in df['Sex']]

# %%
df_features = pd.read_csv('ALL_FEATURES.csv',index_col=0).rename(columns={'ID':'file_name'})
df_new = []
for info, df_info in df.groupby(['ID','Grupa']):
    idx=info[0][-3:]
    if info[1]=='Cardio':  
        if int(idx)<100:
            idx=idx[1:]
        file_name=f'Data_final_cardio_{idx}.csv'
    elif info[1]=='Sport':
        file_name=f'Data_final_{idx}.csv'
    elif info[1]=='Healthy':
        if 'R' in info[0]:
            idx=info[0][-4:]
            file_name=f'Data_final_healthy_{idx}.csv'
        elif int(idx) in [37,47,49,50,51]:
            file_name=f'HH{idx}_supine.csv'
        else:
            file_name=f'HH{idx}.csv'
    df_info['file_name'] = file_name
    df_new.append(df_info)
df_new = pd.concat(df_new)
#%%
df_all = pd.merge(df_new,df_features,on='file_name').drop(['file_name','ID'],axis=1)

#%%
demogr = ['Age','Weight','Height','BMI','Sex']
df_hr = df_all[demogr+hr_features]
df_hr_resp = df_all[demogr+hr_features+resp_features]

X_list = [
    df_hr,
    df_hr_resp,
    df_all.drop('Grupa',axis=1),
    df_all.drop(['Grupa','Sex','Age','Weight','Height','BMI'],axis=1)
]
X_names = [
    'HR',
    'HR_RESP',
    'ALL',
    'ALL_NO_DEMOGR'
]

# %%
class ModelSKL():

    def __init__(self, model_type, **model_params):
        if model_type == 'LogisticRegression':
            mdl = LogisticRegression(class_weight='balanced',**model_params)
        elif model_type == 'SVC':
            mdl = SVC(class_weight='balanced',**model_params)
        elif model_type == 'RandomForest':
            mdl = RandomForestClassifier(class_weight='balanced',**model_params)
        elif model_type == 'GradientBoosting':
            mdl = GradientBoostingClassifier(**model_params)
        elif model_type == 'GaussianNB':
            mdl = GaussianNB(**model_params)
        elif model_type == 'KNeighbors':
            mdl = KNeighborsClassifier(**model_params)
        elif model_type == 'AdaBoost':
            mdl = AdaBoostClassifier(**model_params)
        elif model_type == 'Bagging':
            mdl = BaggingClassifier(**model_params)
        elif model_type == 'DecisionTree':
            mdl = DecisionTreeClassifier(class_weight='balanced',**model_params)
        elif model_type == 'MultinomialNB':
            mdl = MultinomialNB(**model_params)
        elif model_type == 'Ridge':
            mdl = RidgeClassifier(class_weight='balanced',**model_params)
        elif model_type == 'ExtraTrees':
            mdl = ExtraTreesClassifier(class_weight='balanced',**model_params)
        elif model_type == 'LinearSVC':
            mdl = LinearSVC(class_weight='balanced',**model_params)
        elif model_type == 'SGDClassifier':
            mdl = SGDClassifier(class_weight='balanced',**model_params)
        elif model_type == 'XGBClassifier':
            mdl = xgb.XGBClassifier(**model_params)
        # TODO: Add more models as needed
        self.model = mdl
        self.params = {}
        for key in model_params:
            self.params[key] = model_params[key]

    def fit(self, X_train_scaled, y_train, folder_name, idx, **fitting_params):

        self.model.fit(X_train_scaled, y_train, **fitting_params)

        joblib.dump(self.model, f'{folder_name}/model_{idx}.joblib')
        for key in fitting_params:
            self.params[key] = fitting_params[key]

    def predict(self, X):
        return self.model.predict(X)
    
    def finished_cv(self, folder_name):
        pass


class ModelMLP():
    def __init__(self, model_type, neurons, batch_norm, dropout, regularization=None, regularization_alpha=0.1):
        if regularization == 'l1':
            kernel_regularizer = tf.keras.regularizers.L1(l1=regularization_alpha)
        elif regularization == 'l2':
            kernel_regularizer = tf.keras.regularizers.L2(l2=regularization_alpha)
        else:
            kernel_regularizer = None

        mdl = Sequential()
        for neur in neurons:
            mdl.add(Dense(neur, activation='elu', kernel_regularizer=kernel_regularizer))
            if batch_norm:
                mdl.add(BatchNormalization())
            if dropout > 0:
                mdl.add(Dropout(dropout))
        # Change to 3 neurons for 3 classes and softmax activation
        mdl.add(Dense(3, activation='softmax'))  
        self.model = mdl
        self.params = {
            'neurons': str(neurons)[1:-1],
            'batch_norm': batch_norm,
            'dropout': dropout,
            'regularization': regularization,
            'regularization_alpha': regularization_alpha
        }

    def fit(self, X_train_scaled, y_train, X_test_scaled, y_test, folder_name, idx, learning_rate, epochs, batch_size, decay_steps, decay_rate):
        opt = optimizers.Nadam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',  # Changed to categorical_crossentropy
            metrics=[
                'accuracy',
                metrics.AUC(name='auc'),  # AUC metric
                metrics.Precision(name='precision'),  # Precision metric
                metrics.Recall(name='recall')  # Recall metric
            ]  # Changed to accuracy
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        hist = self.model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=epochs, batch_size=batch_size, callbacks=[lr_callback], verbose=False)

        self.model.save(f'{folder_name}/model_{idx}.h5')

        # Update metrics plotting
        for metric in ['loss', 'accuracy', 'auc', 'precision', 'recall']:
            plt.figure(f'{folder_name}_{metric}')
            plt.plot(hist.history[metric],alpha=0.5)
            plt.plot(hist.history[f'val_{metric}'],'--',alpha=0.5)
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            # plt.savefig(f'{folder_name}/{metric}_{idx}.jpg')
            # plt.close()

        self.params['epochs'] = epochs
        self.params['batch_size'] = batch_size
        self.params['learning_rate'] = learning_rate

        return hist
    
    def predict(self, X):
        return self.model.predict(X)

    def finished_cv(self, folder_name):
        for metric in ['loss', 'accuracy', 'auc', 'precision', 'recall']:
            fig = plt.figure(f'{folder_name}_{metric}')
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            fig.savefig(f'{folder_name}/{metric}.jpg')


def encode_and_binarize(y_train, y_test, class_labels):
    encoder = LabelEncoder()
    encoder.fit(class_labels)
    y_train_encoded = encoder.transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    n_classes = len(class_labels)
    y_train_binarized = label_binarize(y_train_encoded, classes=range(n_classes))
    y_test_binarized = label_binarize(y_test_encoded, classes=range(n_classes))

    return y_train_encoded, y_test_encoded, y_train_binarized, y_test_binarized

def score(model, X):
    try:
        y_score = model.predict_proba(X)
    except:
        y_score = model.decision_function(X)
    return y_score

def plot_average_roc(n_classes, title, folder_name, y_test_binarized_list, y_score_list, class_labels):

    y_test_binarized = np.concatenate(y_test_binarized_list)
    y_score = np.concatenate(y_score_list)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    fig = plt.figure()
    colors = ['red', 'blue', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    # plt.show()
    fig.savefig(f'{folder_name}/{title}.jpg')

    auc_value = np.mean(pd.Series(roc_auc))
    return auc_value


def plot_confusion_matrix(cm,title,folder_name,class_labels):
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",cbar=False)
    plt.title(title)
    plt.xticks([0.5,1.5,2.5], class_labels)
    plt.yticks([0.5,1.5,2.5], class_labels)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    # plt.show()
    fig.savefig(f'{folder_name}/{title}.jpg')


def run_cv(X, y, signals, model_type, model_class, model_params, fitting_params, dataset_version, n_features, feature_selection):
    n_splits=10
    kf = StratifiedKFold(n_splits=n_splits, random_state=123, shuffle=True)

    folder_name = f'Artifacts/Artifacts_{model_type}_{datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]}'
    os.makedirs(folder_name)
    
    y_hat_test_all = np.array([])
    y_test_all = np.array([])
    y_hat_train_all = np.array([])
    y_train_all = np.array([])
    accuracy_train = []
    accuracy_test = []
    precision_train = []
    precision_test = []
    recall_train = []
    recall_test = []
    f1_train = []
    f1_test = []
    mcc_train = []
    mcc_test = []
    cumulative_cm_train = None
    cumulative_cm_test = None
    class_labels = ["Cardio", "Healthy", "Sport"]
    y_train_binarized_list = []
    y_train_score_list = []
    y_test_binarized_list = []
    y_test_score_list = []

    for idx, (train_index, test_index) in tqdm(enumerate(kf.split(X, y)),total=n_splits):

        X_train, X_test  =  X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test  =  y.iloc[train_index,], y.iloc[test_index,]
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)    

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

        mdl = model_class(model_type, **model_params)
        smote = SMOTE(random_state=42,sampling_strategy={'Cardio':200,'Healthy':200,'Sport':200})
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        y_train_encoded, y_test_encoded, y_train_binarized, y_test_binarized = encode_and_binarize(y_train, y_test, class_labels)

        
        if model_class.__name__=='ModelMLP':
            enc = OneHotEncoder(handle_unknown='ignore').fit(y_train_encoded.reshape(-1, 1))
            y_train_encoded = enc.transform(y_train_encoded.reshape(-1, 1)).toarray()
            y_test_encoded = enc.transform(y_test_encoded.reshape(-1, 1)).toarray()
            mdl.fit(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, folder_name, idx, **fitting_params)
        else:
            mdl.fit(X_train_scaled, y_train_encoded, folder_name, idx, **fitting_params)

        y_hat_train = mdl.predict(X_train_scaled)
        y_hat_test = mdl.predict(X_test_scaled)
        if model_class.__name__=='ModelMLP':
            y_hat_train = np.argmax(y_hat_train, axis=1)
            y_hat_test = np.argmax(y_hat_test, axis=1)
            y_train_encoded, y_test_encoded, y_train_binarized, y_test_binarized = encode_and_binarize(y_train, y_test, class_labels)
        y_hat_test_all = np.concatenate([y_hat_test_all, y_hat_test.reshape(-1)])
        y_test_all = np.concatenate([y_test_all, y_test_encoded])
        y_hat_train_all = np.concatenate([y_hat_train_all, y_hat_train.reshape(-1)])
        y_train_all = np.concatenate([y_train_all, y_train_encoded])
        accuracy_train.append(accuracy_score(y_train_encoded, y_hat_train))
        accuracy_test.append(accuracy_score(y_test_encoded, y_hat_test))
        precision_train.append(precision_score(y_train_encoded, y_hat_train, average='macro'))
        precision_test.append(precision_score(y_test_encoded, y_hat_test, average='macro'))
        recall_train.append(recall_score(y_train_encoded, y_hat_train, average='macro'))
        recall_test.append(recall_score(y_test_encoded, y_hat_test, average='macro'))
        f1_train.append(f1_score(y_train_encoded, y_hat_train, average='macro'))
        f1_test.append(f1_score(y_test_encoded, y_hat_test, average='macro'))
        mcc_train.append(matthews_corrcoef(y_train_encoded, y_hat_train))
        mcc_test.append(matthews_corrcoef(y_test_encoded, y_hat_test))
        cm_train = confusion_matrix(y_train_encoded, y_hat_train)
        cm_test = confusion_matrix(y_test_encoded, y_hat_test)

        #ROC/AUC
        if model_class.__name__=='ModelMLP':
            y_score_train = mdl.predict(X_train_scaled)
            y_score_test = mdl.predict(X_test_scaled)
        else:
            y_score_train = score(mdl.model, X_train_scaled)
            y_score_test = score(mdl.model, X_test_scaled)

        y_train_binarized_list.append(y_train_binarized)
        y_train_score_list.append(y_score_train)

        y_test_binarized_list.append(y_test_binarized)
        y_test_score_list.append(y_score_test)

        # Sum the confusion matrices
        if cumulative_cm_train is None:
            cumulative_cm_train = cm_train
        else:
            cumulative_cm_train += cm_train

        if cumulative_cm_test is None:
            cumulative_cm_test = cm_test
        else:
            cumulative_cm_test += cm_test
    mdl.finished_cv(folder_name)
    auc_train = plot_average_roc(len(class_labels), 'Train_roc', folder_name, y_train_binarized_list, y_train_score_list, class_labels)
    auc_test = plot_average_roc(len(class_labels), 'Test_roc', folder_name, y_test_binarized_list, y_test_score_list, class_labels)

    # Plot cumulative confusion matrices
    plot_confusion_matrix(cumulative_cm_train, 'Train_cm', folder_name, class_labels)
    plot_confusion_matrix(cumulative_cm_test, 'Test_cm', folder_name, class_labels)
    
    res_train = pd.DataFrame({
        'Accuracy': accuracy_train,
        'Precision': precision_train,
        'Recall': recall_train,
        'F1': f1_train,
        'MCC': mcc_train,
        'AUC': auc_train
    })
    res_test = pd.DataFrame({
        'Accuracy': accuracy_test,
        'Precision': precision_test,
        'Recall': recall_test,
        'F1': f1_test,
        'MCC': mcc_test,
        'AUC': auc_test
    })
    results  = {'train': res_train, 'test': res_test}
    res_train.to_csv(f'{folder_name}/train_metrics.csv')
    res_test.to_csv(f'{folder_name}/test_metrics.csv')
    pd.DataFrame({
        'y_true': y_test_all,
        'y_pred': y_hat_test_all
        }).to_csv(f'{folder_name}/y_true_pred.csv')
    # Logging to Neptune
    params = mdl.params
    params['dataset_version'] = dataset_version
    params['model_type'] = model_type

    run = neptune.init_run(
        project="mrosol5/Cardio-Healthy-Sport",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyYWM1MmYxNy00OWY1LTRhMTUtYTczNS05YjFkODEwYWJjZTgifQ==",
    ) 
    run["parameters"] = params
    for stage in ['train', 'test']:
        for metric in ['Accuracy', 'Recall', 'Precision', 'F1', 'MCC', 'AUC']:
            run[f'{stage}/{metric}'] = results[stage].mean()[metric]
    run["Artifacts"].upload_files(f"{folder_name}/*")  
    run["sys/tags"].add([signals])
    run.stop()
    logging.info(f"Accuracy={results['test'].mean()['Accuracy']}")

    #Deleting all artifacts
    for filename in os.listdir(folder_name):
        filepath = os.path.join(folder_name, filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting {filepath}: {e}")
    os.rmdir(folder_name) 
    plt.close('all')
    print(f'Train accuracy = {np.mean(results["train"]["Accuracy"])}')
    print(f'Test accuracy = {np.mean(results["test"]["Accuracy"])}')
    return y_hat_test_all, y_test_all, results

y = df_all['Grupa']
done_model = []
#%% LR

dataset_version = '140524'
model_type = 'LogisticRegression'
logging.info(f'Starting {model_type}')

for C in [0.01,0.1,0.5,0.75,1.0,1.25,1.5,5.0,10,25,100]: 
    for penalty in ['l1', 'l2','elasticnet']:
        model_params = {
            'C': C, 
            'solver':'saga',
            'penalty': penalty,
            'l1_ratio':0.5
            }
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')
#%% Decision tree
dataset_version = '140524'
model_type = 'DecisionTree'
logging.info(f'Starting {model_type}')
for max_depth in [3, 5, 10, None]:
    for criterion in ['gini', 'entropy']:
        model_params = {'max_depth': max_depth, 'criterion': criterion}
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')
#%% Extra trees
dataset_version = '140524'
model_type = 'ExtraTrees'
logging.info(f'Starting {model_type}')
for n_estimators in [50, 100, 200]:
    for max_depth in [3, 10, None]:
        model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')

#%% RF
dataset_version = '140524'
model_type = 'RandomForest'
logging.info(f'Starting {model_type}')
for n_estimators in [10, 100, 300]:
    for max_depth in [3, 10, None]:
        model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#%% SVM
dataset_version = '140524'
model_type = 'SVC'
logging.info(f'Starting {model_type}')
for kernel in ['rbf', 'linear', 'poly']:
    for C in [0.1, 1.0, 10.0]:
        model_params = {'kernel': kernel, 'C': C}
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')

#%% AdaBoost
dataset_version = '140524'
model_type = 'AdaBoost'
logging.info(f'Starting {model_type}')
for n_estimators in [200, 300, 500]: #50, 100, 200
    for learning_rate in [0.001, 0.01]: #0.01, 0.1, 1.0
        model_params = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')

#%% XGBClassifier
dataset_version = '140524'
model_type = 'XGBClassifier'
logging.info(f'Starting {model_type}')
for n_estimators in [25,50,75]: #50, 100, 200
    for learning_rate in [0.05,0.1,0.15]: #0.01, 0.1, 0.2
        model_params = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')

##%% GradientBoosting

dataset_version = '140524'
model_type = 'GradientBoosting'
logging.info(f'Starting {model_type}')
for n_estimators in [10,100,300]:
    for lr in [1.0,0.5,0.1, 0.01]:
        model_params = {
            'n_estimators': n_estimators,
            'learning_rate': lr
            }
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')

##%% Naive Bayes
dataset_version = '140524'
model_type = 'GaussianNB'
logging.info(f'Starting {model_type}')
for var_smoothing in [1e-9, 1e-8, 1e-7]:
    model_params = {'var_smoothing': var_smoothing}
    fitting_params = {}
    for idx in range(len(X_list)):
        X = X_list[idx]
        X_name = X_names[idx]
        _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')

##%% Bagging classifier
dataset_version = '140524'
model_type = 'Bagging'
logging.info(f'Starting {model_type}')

for n_estimators in [10, 50, 100]:
    for max_samples in [0.5, 1.0]:
        model_params = {'n_estimators': n_estimators, 'max_samples': max_samples}
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')

##%% Ridge classifier

dataset_version = '140524'
model_type = 'Ridge'
logging.info(f'Starting {model_type}')
for alpha in [10, 5, 1.0, 0.5, 0.1, 0.01, 0.001]:
    model_params = {'alpha': alpha}
    fitting_params = {}
    for idx in range(len(X_list)):
        X = X_list[idx]
        X_name = X_names[idx]
        _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')

##%% LinearSVC
dataset_version = '140524'
model_type = 'LinearSVC'
for C in [1, 0.1, 0.01]:
    for loss in ['hinge', 'squared_hinge']:
        model_params = {'C': C, 'loss': loss}
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')

##%% KNN
dataset_version = '140524'
model_type = 'KNeighbors'
logging.info(f'Starting {model_type}')
# Define a range for n_neighbors
for n_neighbors in [3, 5, 7, 10]:
    # Define a list of distance metrics
    for metric in ['minkowski', 'euclidean', 'manhattan']:
        # Define model parameters
        model_params = {
            'n_neighbors': n_neighbors,
            'metric': metric
        }
        # Initialize fitting_params as an empty dictionary
        fitting_params = {}

        # Iterate over your datasets
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            # Call your cross-validation function
            _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
done_model.append(model_type)
pd.Series(done_model).to_csv('Done_models.csv')

#%%

dataset_version = '310124_4'
model_type = 'MLP'

# Define different configurations for the neural network
neurons_configurations = [
    [32, 32],
    [64, 64],        
    [128, 64, 32]    
]
dropout_rates = [0.1,0.2,0.5] 

# Regularization configurations
regularization_options = [None, 'l1', 'l2']
regularization_alpha_values = [0.01, 0.1]

for neurons in neurons_configurations:
        for dropout in dropout_rates:
            for regularization in regularization_options:
                for reg_alpha in regularization_alpha_values:
                    for learning_rate in [1e-2,1e-3]: #,1e-4,1e-5
                        for batch_size in [4]:
                            model_params = {
                                'neurons': neurons,
                                'batch_norm': True,
                                'dropout': dropout,
                                'regularization': regularization,
                                'regularization_alpha': reg_alpha
                            }
                            fitting_params = {
                                'learning_rate' : learning_rate, 
                                'epochs': 100, 
                                'batch_size': batch_size, 
                                'decay_steps': 10, 
                                'decay_rate': 0.90 
                            }

                            # Iterate over your datasets
                            for idx in range(len(X_list)):
                                X = X_list[idx]
                                X_name = X_names[idx]
                                _, _, _ = run_cv(X, y, X_name, model_type, ModelMLP, model_params, fitting_params, dataset_version)
