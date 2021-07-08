import pandas as pd
import numpy as np
import sys
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import MSomte as ms
from imblearn.over_sampling import SMOTE
from collections import Counter
import processing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                 "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                 "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                 "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                 "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                 "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                 "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                 "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    df = pd.read_csv("KDDTrain+.csv", header=None, names=col_names)
    df_test = pd.read_csv("KDDTest+_2.csv", header=None, names=col_names)
    df,df_test = processing.change(df,df_test)
    X_test_a = np.array(processing.changetag(df_test))[:, :-1]
    df_X = np.array(df)[:, :-1]
    df_y = np.ravel(np.array(df)[:, -1:])
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.45,random_state= 42)
    df_training = pd.DataFrame(np.column_stack((X_train, y_train)),columns = df.columns.values.tolist())
    df_testing = pd.DataFrame(np.column_stack((X_test, y_test)),columns = df.columns.values.tolist())
    std = MinMaxScaler()
    # todo Normal
    df_Normal = processing.changetag_Normal(df_training)
    df_Normal = std.fit_transform(df_Normal)
    # df_Normal = processing.Standardization(df_Normal)
    df_test_Normal = processing.changetag_Normal(df_testing)
    df_test_Normal = std.fit_transform(df_test_Normal)

    df_Normal_fs ,df_test_Normal_fs,X_test_a_Normalfs = df_Normal[:, 120] , df_test_Normal[:,120],X_test_a[:,120]
    list_Normal = [8, 116, 65, 26, 90, 35, 1, 33 ,25 ,22, 53]
    for i in list_Normal:
        df_Normal_0 = df_Normal[:, i]
        df_Normal_fs = np.column_stack((df_Normal_fs, df_Normal_0))
        df_test_Normal_0 = df_test_Normal[:,i]
        df_test_Normal_fs = np.column_stack((df_test_Normal_fs, df_test_Normal_0))
        X_test_a_0 = X_test_a[:, i]
        X_test_a_Normalfs = np.column_stack((X_test_a_Normalfs, X_test_a_0))
    # df_test_Normal = processing.Standardization(df_test_Normal)
    X = df_Normal_fs  # All rows, omit last column
    y = np.ravel(np.array(df_Normal)[:, -1:])  # All rows, only the last column
    # print(Counter(y))
    #smote
    sm = SMOTE(random_state=42)
    X_train_Normal, y_train_Normal = sm.fit_resample(X,y)
    #msmote
    # R2L_df = df[[each == 1 for each in df['label']]]
    # #for m in range(10000,15000,100):
    # msmote = ms.MSmote(N=m)
    # samples = msmote.fit(R2L_df)
    # df = np.append(pd.DataFrame(df), samples, axis=0)
    # df = pd.DataFrame(df)
    # X_train = np.array(df)[:, :-1]  # All rows, omit last column
    # y_train = np.ravel(np.array(df)[:, -1:])  # All rows, only the last column
    #normal
    # X_train_Normal = X
    # y_train_Normal = y
    # print(Counter(y))

    # X_train_Normal = std.fit_transform(X_train_Normal)
    X_test_Normal = df_test_Normal_fs  # All rows, omit last column
    y_test_Normal = np.ravel(np.array(df_test_Normal)[:, -1:])  # All rows, only the last column
    # print(Counter(y_test_Normal))
    #
    for i in range(10,210,10):
        rfc = RandomForestClassifier(n_estimators=i, random_state=42, max_depth=4,)
        rfc.fit(X_train_Normal, y_train_Normal.astype('int'))
        y_pred_Normal = rfc.score(X_test_Normal, y_test_Normal.astype('int'))
        y_pred_Normal_t = rfc.fit(X_train_Normal, y_train_Normal.astype('int')).predict(X_test_Normal)
        print("%d:%.10f"%(i,y_pred_Normal))
        print(Counter(y_test_U2R))
        print(Counter(y_pred_t_U2R))