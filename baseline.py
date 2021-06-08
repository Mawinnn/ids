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
from sklearn.metrics import f1_score


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

    df = processing.changetag(df)
    X = np.array(df)[:, :-1]  # All rows, omit last column
    y = np.ravel(np.array(df)[:, -1:])  # All rows, only the last column
    # smote
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X, y.astype('int'))
    std = MinMaxScaler()
    X_train = std.fit_transform(X_train)

    df_test = processing.changetag(df_test)
    X_test = np.array(df_test)[:, :-1]  # All rows, omit last column
    y_test = np.ravel(np.array(df_test)[:, -1:])  # All rows, only the last column
    X_test = std.fit_transform(X_test)
    # RF
    # for i in range(1,20):
    rfc = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=4, )
    rfc.fit(X_train, y_train.astype('int'))
    print( rfc.score(X_test,y_test.astype('int')))
    y_pre = rfc.fit(X_train, y_train.astype('int')).predict(X_test)
    print(f1_score(y_test,y_pre,average='macro'))

    #SVM
    # svm_model = SVC(kernel='rbf', C=1000, gamma=0.001)  # 最佳模型
    # svm_model.fit(X_train, y_train.astype('int'))
    # print (svm_model.score(X_test,y_test.astype('int')))