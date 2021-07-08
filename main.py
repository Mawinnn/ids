import pandas as pd
import numpy as np
import sys
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import MSomte as ms
from imblearn.over_sampling import SMOTE
from collections import Counter
import processing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix

def heatmap(confusion_matrix,X_n,Y_n):
    fig, ax = plt.subplots()
    # 将元组分解为fig和ax两个变量
    im = ax.imshow(confusion_matrix)
    # 显示图片

    ax.set_xticks(np.arange(len(X_n)))
    # 设置x轴刻度间隔
    ax.set_yticks(np.arange(len(Y_n)))
    # 设置y轴刻度间隔
    ax.set_xticklabels(X_n)
    # 设置x轴标签'''
    ax.set_yticklabels(Y_n)
    # 设置y轴标签'''

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # 设置标签 旋转45度 ha有三个选择：right,center,left（对齐方式）

    for i in range(len(X_n)):
        for j in range(len(Y_n)):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    # 设置题目
    fig.tight_layout()  # 自动调整子图参数,使之填充整个图像区域。
    plt.show()  # 图像展示

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
    # for m in range(15,90,5):
    #     n = m/100
    #     print("---------%.2f---------"%n)

    for a in range(8,9):
        print("---------------------%d--------------------"%a)
        list =[a >>d & 1 for d in range(5)][::-1]
        print(list)
        df = pd.read_csv("KDDTrain+.csv", header=None, names=col_names)
        df_test = pd.read_csv("KDDTest+_2.csv", header=None, names=col_names)
        X_test_b = np.array(processing.changetag(df_test))[:, :-1]
        df_X = np.array(df)[:, :-1]
        df_y = np.ravel(np.array(df)[:, -1:])

        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.45, random_state=42)
        df_training = pd.DataFrame(np.column_stack((X_train, y_train)), columns=col_names)
        df_testing = pd.DataFrame(np.column_stack((X_test, y_test)), columns=col_names)
        std = MinMaxScaler()
    # todo Normal
        df_Normal = processing.changetag_Normal(df_training)
        df_Normal = std.fit_transform(df_Normal)
        # df_Normal = processing.Standardization(df_Normal)
        df_test_Normal = processing.changetag_Normal(df_testing)
        df_test_Normal = std.fit_transform(df_test_Normal)
        if (list[0]==1):
            df_Normal_fs, df_test_Normal_fs, X_test_a_Normalfs = df_Normal[:, 11], df_test_Normal[:, 11], X_test_b[:, 11]
            list_Normal = [3, 29, 25, 2, 36, 38, 28, 4]
            for i in list_Normal:
                df_Normal_0 = df_Normal[:, i]
                df_Normal_fs = np.column_stack((df_Normal_fs, df_Normal_0))
                df_test_Normal_0 = df_test_Normal[:, i]
                df_test_Normal_fs = np.column_stack((df_test_Normal_fs, df_test_Normal_0))
                X_test_a_0 = X_test_b[:, i]
                X_test_a_Normalfs = np.column_stack((X_test_a_Normalfs, X_test_a_0))
            # df_test_Normal = processing.Standardization(df_test_Normal)
            X = np.array(df_Normal_fs)  # All rows, omit last column
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

            X_train_Normal = std.fit_transform(X_train_Normal)
            X_test_Normal = np.array(df_test_Normal_fs)  # All rows, omit last column
            y_test_Normal = np.ravel(np.array(df_test_Normal)[:, -1:])  # All rows, only the last column
            # print(Counter(y_test_Normal))

            # for i in range(1,20):
            rfc = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=9,)
            rfc.fit(X_train_Normal, y_train_Normal.astype('int'))
            y_pred_Normal = rfc.score(X_test_Normal, y_test_Normal.astype('int'))
            y_pred_Normal_t = rfc.fit(X_train_Normal, y_train_Normal.astype('int')).predict(X_test_Normal)
            # print("%d:%.10f"%(i,y_pred_Normal))
            # print(Counter(y_pred_Normal_t))
            #print(y_pred_t)
            X_train_1 = rfc.predict_proba(X_test_Normal)
            X_test_a_Normalfs = std.fit_transform(X_test_a_Normalfs)
            X_test_NN = rfc.predict_proba(X_test_a_Normalfs)
        else:
            X = np.array(df_Normal)[:, :-1]  # All rows, omit last column
            y = np.ravel(np.array(df_Normal)[:, -1:])  # All rows, only the last column
            # print(Counter(y))
            # smote
            sm = SMOTE(random_state=42)
            X_train_Normal, y_train_Normal = sm.fit_resample(X, y.astype('int'))
            # msmote
            # Normal_df =pd.DataFrame(df_Normal,columns=col_names)
            # Normal_df = Normal_df[[each == 1 for each in Normal_df['label']]]
            # #for m in range(10000,15000,100):
            # msmote = ms.MSmote(N=200)
            # samples = msmote.fit(Normal_df)
            # df_Normal = pd.DataFrame(np.append(pd.DataFrame(df_Normal), samples, axis=0))
            # X_train_Normal = np.array(df_Normal)[:, :-1]  # All rows, omit last column
            # y_train_Normal = np.ravel(np.array(df_Normal)[:, -1:])  # All rows, only the last column
            # normal
            # X_train_Normal = X
            # y_train_Normal = y
            # print(Counter(y))
            X_train_Normal = std.fit_transform(X_train_Normal)
            X_test_Normal = np.array(df_test_Normal)[:, :-1]  # All rows, omit last column
            y_test_Normal = np.ravel(np.array(df_test_Normal)[:, -1:])  # All rows, only the last column
            # print(Counter(y_test_Normal))

            # for i in range(1,20):
            rfc = RandomForestClassifier(n_estimators=130, random_state=42, max_depth=15, )
            rfc.fit(X_train_Normal, y_train_Normal.astype('int'))
            y_pred_Normal = rfc.score(X_test_Normal, y_test_Normal.astype('int'))
            y_pred_t_Normal = rfc.fit(X_train_Normal, y_train_Normal.astype('int')).predict(X_test_Normal)
            # print("%d:%.10f"%(i,y_pred_Normal))
            # print(Counter(y_test_Normal))
            # print(Counter(y_pred_t_Normal))
            # print(Counter(y_pred_t_Normal))
            # print(y_pred_t_Normal)

            X_train_1 = rfc.predict_proba(X_test_Normal)
            X_test_a_Normal = std.fit_transform(X_test_b)
            X_test_NN = rfc.predict_proba(X_test_a_Normal)

        # todo Dos
        df_training = pd.DataFrame(np.column_stack((X_train, y_train)), columns=col_names)
        df_testing = pd.DataFrame(np.column_stack((X_test, y_test)), columns=col_names)
        df_Dos = processing.changetag_Dos(df_training)
        # df_Dos = std.fit_transform(df_Dos)
        # df_Dos = processing.Standardization(df_Dos)
        df_test_Dos = processing.changetag_Dos(df_testing)
        df_test_Dos = std.fit_transform(df_test_Dos)
        if(list[1] == 1):
            df_Dos_fs, df_test_Dos_fs, X_test_a_Dosfs = np.array(df_Dos)[:, 3], df_test_Dos[:, 3], X_test_b[:, 3]
            list_Dos = [25, 11, 29, 38, 36, 24, 4, 37 ,30 ,28 ,2]
            for i in list_Dos:
                df_Dos_0 = np.array(df_Dos)[:, i]
                df_Dos_fs = np.column_stack((df_Dos_fs, df_Dos_0))
                df_test_Dos_0 = df_test_Dos[:, i]
                df_test_Dos_fs = np.column_stack((df_test_Dos_fs, df_test_Dos_0))
                X_test_a_1 = X_test_b[:, i]
                X_test_a_Dosfs = np.column_stack((X_test_a_Dosfs, X_test_a_1))
            # df_test_Dos = processing.Standardization(df_test_Dos)
            X = np.array(df_Dos_fs)  # All rows, omit last column
            y = np.ravel(np.array(df_Dos)[:, -1:])  # All rows, only the last column
            # print(Counter(y))
            #smote
            sm = SMOTE(random_state=42)
            X_train_Dos, y_train_Dos = sm.fit_resample(X,y.astype('int'))
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
            # X_train_Dos = X
            # y_train_Dos = y
            # print(Counter(y))
            X_train_Dos = std.fit_transform(X_train_Dos)
            X_test_Dos = np.array(df_test_Dos_fs)  # All rows, omit last column
            y_test_Dos = np.ravel(np.array(df_test_Dos)[:, -1:])  # All rows, only the last column
            # print(Counter(y_test_Dos))

            # for i in range(1,20):
            rfc = RandomForestClassifier(n_estimators=110, random_state=42, max_depth=15,)
            rfc.fit(X_train_Dos, y_train_Dos.astype('int'))
            y_pred_Dos = rfc.score(X_test_Dos, y_test_Dos.astype('int'))
            y_pred_t_Dos = rfc.fit(X_train_Dos, y_train_Dos.astype('int')).predict(X_test_Dos)
            # print("%d:%.10f"%(i,y_pred_Dos))
            # print(Counter(y_pred_t_Dos))
            #print(y_pred_t)
            X_train_1 = np.column_stack((X_train_1, rfc.predict_proba(X_test_Dos)))
            X_test_a_Dosfs = std.fit_transform(X_test_a_Dosfs)
            X_test_NN = np.column_stack((X_test_NN,rfc.predict_proba(X_test_a_Dosfs)))
        else:
            X = np.array(df_Dos)[:, :-1]  # All rows, omit last column
            y = np.ravel(np.array(df_Dos)[:, -1:])  # All rows, only the last column
            # print(Counter(y))
            # smote
            sm = SMOTE(random_state=42)
            X_train_Dos, y_train_Dos = sm.fit_resample(X, y.astype('int'))
            # msmote
            # Dos_df =pd.DataFrame(df_Dos,columns=col_names)
            # Dos_df = Dos_df[[each == 1 for each in Dos_df['label']]]
            # #for m in range(10000,15000,100):
            # msmote = ms.MSmote(N=200)
            # samples = msmote.fit(Dos_df)
            # df_Dos = pd.DataFrame(np.append(pd.DataFrame(df_Dos), samples, axis=0))
            # X_train_Dos = np.array(df_Dos)[:, :-1]  # All rows, omit last column
            # y_train_Dos = np.ravel(np.array(df_Dos)[:, -1:])  # All rows, only the last column
            # normal
            # X_train_Dos = X
            # y_train_Dos = y
            # print(Counter(y))
            X_train_Dos = std.fit_transform(X_train_Dos)
            X_test_Dos = np.array(df_test_Dos)[:, :-1]  # All rows, omit last column
            y_test_Dos = np.ravel(np.array(df_test_Dos)[:, -1:])  # All rows, only the last column
            # print(Counter(y_test_Dos))

            # for i in range(1,20):
            rfc = RandomForestClassifier(n_estimators=40, random_state=42, max_depth=13, )
            rfc.fit(X_train_Dos, y_train_Dos.astype('int'))
            y_pred_Dos = rfc.score(X_test_Dos, y_test_Dos.astype('int'))
            y_pred_t_Dos = rfc.fit(X_train_Dos, y_train_Dos.astype('int')).predict(X_test_Dos)
            # print("%d:%.10f"%(i,y_pred_Dos))
            # print(Counter(y_test_Dos))
            # print(Counter(y_pred_t_Dos))
                # print(Counter(y_pred_t_Dos))
            # print(y_pred_t_Dos)
            X_train_1 = np.column_stack((X_train_1, rfc.predict_proba(X_test_Dos)))
            X_test_a_Dos = std.fit_transform(X_test_b)
            X_test_NN = np.column_stack((X_test_NN, rfc.predict_proba(X_test_a_Dos)))


        # todo Probe
        df_training = pd.DataFrame(np.column_stack((X_train, y_train)), columns=col_names)
        df_testing = pd.DataFrame(np.column_stack((X_test, y_test)), columns=col_names)
        df_Probe = processing.changetag_Probe(df_training)
        # df_Probe = std.fit_transform(df_Probe)
        # df_Probe = processing.Standardization(df_Probe)
        df_test_Probe = processing.changetag_Probe(df_testing)
        df_test_Probe = std.fit_transform(df_test_Probe)
        # df_test_Probe = processing.Standardization(df_test_Probe)
        if (list[2]== 1):
            df_Probe_fs, df_test_Probe_fs, X_test_a_Probefs = np.array(df_Probe)[:, 11], df_test_Probe[:, 11], X_test_b[:, 11]
            list_Probe = [2, 3, 36, 35, 38, 4, 30, 40, 29, 23, 5, 37]
            for i in list_Probe:
                df_Probe_0 = np.array(df_Probe)[:, i]
                df_Probe_fs = np.column_stack((df_Probe_fs, df_Probe_0))
                df_test_Probe_0 = df_test_Probe[:, i]
                df_test_Probe_fs = np.column_stack((df_test_Probe_fs, df_test_Probe_0))
                X_test_a_1 = X_test_b[:, i]
                X_test_a_Probefs = np.column_stack((X_test_a_Probefs, X_test_a_1))
            X = np.array(df_Probe_fs)  # All rows, omit last column
            y = np.ravel(np.array(df_Probe)[:, -1:])  # All rows, only the last column
            # print(Counter(y))
            # smote
            sm = SMOTE(random_state=42)
            X_train_Probe, y_train_Probe = sm.fit_resample(X, y.astype('int'))
            # msmote
            # R2L_df = df[[each == 1 for each in df['label']]]
            # #for m in range(10000,15000,100):
            # msmote = ms.MSmote(N=m)
            # samples = msmote.fit(R2L_df)
            # df = np.append(pd.DataFrame(df), samples, axis=0)
            # df = pd.DataFrame(df)
            # X_train = np.array(df)[:, :-1]  # All rows, omit last column
            # y_train = np.ravel(np.array(df)[:, -1:])  # All rows, only the last column
            # normal
            # X_train_Probe = X
            # y_train_Probe = y
            # print(Counter(y))
            X_train_Probe = std.fit_transform(X_train_Probe)
            X_test_Probe = np.array(df_test_Probe_fs)  # All rows, omit last column
            y_test_Probe = np.ravel(np.array(df_test_Probe)[:, -1:])  # All rows, only the last column
            # print(Counter(y_test_Probe))

            # for i in range(60,80):
            rfc = RandomForestClassifier(n_estimators=70, random_state=42, max_depth=19, )
            rfc.fit(X_train_Probe, y_train_Probe.astype('int'))
            y_pred_Probe = rfc.score(X_test_Probe, y_test_Probe.astype('int'))
            y_pred_t_Probe = rfc.fit(X_train_Probe, y_train_Probe.astype('int')).predict(X_test_Probe)
            # print("%d:%.10f"%(i,y_pred_Probe))
            # print(Counter(y_pred_t_Probe))
            # print(y_pred_t)
            X_train_1 = np.column_stack((X_train_1, rfc.predict_proba(X_test_Probe)))
            X_test_a_Probefs = std.fit_transform(X_test_a_Probefs)
            X_test_NN = np.column_stack((X_test_NN,rfc.predict_proba(X_test_a_Probefs)))
        else :
            X = np.array(df_Probe)[:, :-1]  # All rows, omit last column
            y = np.ravel(np.array(df_Probe)[:, -1:])  # All rows, only the last column
            # print(Counter(y))
            # smote
            sm = SMOTE(random_state=42)
            X_train_Probe, y_train_Probe = sm.fit_resample(X, y.astype('int'))
            # msmote
            # Probe_df =pd.DataFrame(df_Probe,columns=col_names)
            # Probe_df = Probe_df[[each == 1 for each in Probe_df['label']]]
            # #for m in range(10000,15000,100):
            # msmote = ms.MSmote(N=200)
            # samples = msmote.fit(Probe_df)
            # df_Probe = pd.DataFrame(np.append(pd.DataFrame(df_Probe), samples, axis=0))
            # X_train_Probe = np.array(df_Probe)[:, :-1]  # All rows, omit last column
            # y_train_Probe = np.ravel(np.array(df_Probe)[:, -1:])  # All rows, only the last column
            # normal
            # X_train_Probe = X
            # y_train_Probe = y
            # print(Counter(y))
            X_train_Probe = std.fit_transform(X_train_Probe)
            X_test_Probe = np.array(df_test_Probe)[:, :-1]  # All rows, omit last column
            y_test_Probe = np.ravel(np.array(df_test_Probe)[:, -1:])  # All rows, only the last column
            # print(Counter(y_test_Probe))

            # for i in range(1,20):
            rfc = RandomForestClassifier(n_estimators=90, random_state=42, max_depth=14, )
            rfc.fit(X_train_Probe, y_train_Probe.astype('int'))
            y_pred_Probe = rfc.score(X_test_Probe, y_test_Probe.astype('int'))
            y_pred_t_Probe = rfc.fit(X_train_Probe, y_train_Probe.astype('int')).predict(X_test_Probe)
            # print("%d:%.10f"%(i,y_pred_Probe))
            # print(Counter(y_test_Probe))
            # print(Counter(y_pred_t_Probe))
                # print(Counter(y_pred_t_Probe))
            # print(y_pred_t_Probe)
            X_train_1 = np.column_stack((X_train_1, rfc.predict_proba(X_test_Probe)))
            X_test_a_Probe = std.fit_transform(X_test_b)
            X_test_NN = np.column_stack((X_test_NN, rfc.predict_proba(X_test_a_Probe)))

        # todo R2L
        df_training = pd.DataFrame(np.column_stack((X_train, y_train)), columns=col_names)
        df_testing = pd.DataFrame(np.column_stack((X_test, y_test)), columns=col_names)
        df_R2L = processing.changetag_R2L(df_training)
        # df_R2L = std.fit_transform(df_R2L)
        # df_R2L = processing.Standardization(df_R2L)
        df_test_R2L = processing.changetag_R2L(df_testing)
        df_test_R2L = std.fit_transform(df_test_R2L)
        # df_test_R2L = processing.Standardization(df_test_R2L)
        if (list[3] == 1):
            df_R2L_fs, df_test_R2L_fs, X_test_a_R2Lfs = np.array(df_R2L)[:, 2], df_test_R2L[:, 2], X_test_b[:, 2]
            list_R2L = [21, 11, 29, 23, 25, 1, 4, 40, 9, 28, 35]
            for i in list_R2L:
                df_R2L_0 = np.array(df_R2L)[:, i]
                df_R2L_fs = np.column_stack((df_R2L_fs, df_R2L_0))
                df_test_R2L_0 = df_test_R2L[:, i]
                df_test_R2L_fs = np.column_stack((df_test_R2L_fs, df_test_R2L_0))
                X_test_a_1 = X_test_b[:, i]
                X_test_a_R2Lfs = np.column_stack((X_test_a_R2Lfs, X_test_a_1))

            X = np.array(df_R2L_fs)  # All rows, omit last column
            y = np.ravel(np.array(df_R2L)[:, -1:])  # All rows, only the last column
            # print(Counter(y))
            # smote
            sm = SMOTE(random_state=42)
            X_train_R2L, y_train_R2L = sm.fit_resample(X, y.astype('int'))
            # msmote
            # R2L_df =pd.DataFrame(df_R2L,columns=col_names)
            # R2L_df = R2L_df[[each == 1 for each in R2L_df['label']]]
            # #for m in range(10000,15000,100):
            # msmote = ms.MSmote(N=200)
            # samples = msmote.fit(R2L_df)
            # df_R2L = pd.DataFrame(np.append(pd.DataFrame(df_R2L), samples, axis=0))
            # X_train_R2L = np.array(df_R2L)[:, :-1]  # All rows, omit last column
            # y_train_R2L = np.ravel(np.array(df_R2L)[:, -1:])  # All rows, only the last column
            # normal
            # X_train_R2L = X
            # y_train_R2L = y
            # print(Counter(y))
            X_train_R2L = std.fit_transform(X_train_R2L)
            X_test_R2L = np.array(df_test_R2L_fs)  # All rows, omit last column
            y_test_R2L = np.ravel(np.array(df_test_R2L)[:, -1:])  # All rows, only the last column
            # print(Counter(y_test_R2L))

            # for i in range(121,140):
            rfc = RandomForestClassifier(n_estimators=121, random_state=42, max_depth=23, )
            rfc.fit(X_train_R2L, y_train_R2L.astype('int'))
            y_pred_R2L = rfc.score(X_test_R2L, y_test_R2L.astype('int'))
            y_pred_t_R2L = rfc.fit(X_train_R2L, y_train_R2L.astype('int')).predict(X_test_R2L)
            # print("%d:%.10f"%(i,y_pred_R2L))
            # print(Counter(y_pred_t_R2L))
            #print(y_pred_t_R2L)
            X_train_1 = np.column_stack((X_train_1, rfc.predict_proba(X_test_R2L)))
            X_test_a_R2Lfs = std.fit_transform(X_test_a_R2Lfs)
            X_test_NN = np.column_stack((X_test_NN,rfc.predict_proba(X_test_a_R2Lfs)))

        else:
            X = np.array(df_R2L)[:, :-1]  # All rows, omit last column
            y = np.ravel(np.array(df_R2L)[:, -1:])  # All rows, only the last column
            # print(Counter(y))
            # smote
            sm = SMOTE(random_state=42)
            X_train_R2L, y_train_R2L = sm.fit_resample(X, y.astype('int'))
            # msmote
            # R2L_df =pd.DataFrame(df_R2L,columns=col_names)
            # R2L_df = R2L_df[[each == 1 for each in R2L_df['label']]]
            # #for m in range(10000,15000,100):
            # msmote = ms.MSmote(N=200)
            # samples = msmote.fit(R2L_df)
            # df_R2L = pd.DataFrame(np.append(pd.DataFrame(df_R2L), samples, axis=0))
            # X_train_R2L = np.array(df_R2L)[:, :-1]  # All rows, omit last column
            # y_train_R2L = np.ravel(np.array(df_R2L)[:, -1:])  # All rows, only the last column
            # normal
            # X_train_R2L = X
            # y_train_R2L = y
            # print(Counter(y))
            X_train_R2L = std.fit_transform(X_train_R2L)
            X_test_R2L = np.array(df_test_R2L)[:, :-1]  # All rows, omit last column
            y_test_R2L = np.ravel(np.array(df_test_R2L)[:, -1:])  # All rows, only the last column
            # print(Counter(y_test_R2L))

            # for i in range(1,20):
            rfc = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=18, )
            rfc.fit(X_train_R2L, y_train_R2L.astype('int'))
            y_pred_R2L = rfc.score(X_test_R2L, y_test_R2L.astype('int'))
            y_pred_t_R2L = rfc.fit(X_train_R2L, y_train_R2L.astype('int')).predict(X_test_R2L)
            # print("%d:%.10f"%(i,y_pred_R2L))
            # print(Counter(y_test_R2L))
            # print(Counter(y_pred_t_R2L))
                # print(Counter(y_pred_t_R2L))
                # print(y_pred_t_R2L)
            X_train_1 = np.column_stack((X_train_1, rfc.predict_proba(X_test_R2L)))
            X_test_a_R2L = std.fit_transform(X_test_b)
            X_test_NN = np.column_stack((X_test_NN, rfc.predict_proba(X_test_a_R2L)))

        # todo U2R
        df_training = pd.DataFrame(np.column_stack((X_train, y_train)), columns=col_names)
        df_testing = pd.DataFrame(np.column_stack((X_test, y_test)), columns=col_names)
        df_U2R = processing.changetag_U2R(df_training)
        # df_U2R = processing.Standardization(df_U2R)
        df_test_U2R = std.fit_transform(processing.changetag_U2R(df_testing))
        df_test_U2R = std.fit_transform(df_test_U2R)
        if (list[4] == 1):
            df_U2R_fs, df_test_U2R_fs, X_test_a_U2Rfs = np.array(df_U2R)[:, 3], df_test_U2R[:, 3], X_test_b[:, 3]
            list_U2R = [13, 2, 23, 11, 35, 34, 25, 37, 30, 36, 27]
            for i in list_U2R:
                df_U2R_0 = np.array(df_U2R)[:, i]
                df_U2R_fs = np.column_stack((df_U2R_fs, df_U2R_0))
                df_test_U2R_0 = df_test_U2R[:, i]
                df_test_U2R_fs = np.column_stack((df_test_U2R_fs, df_test_U2R_0))
                X_test_a_1 = X_test_b[:, i]
                X_test_a_U2Rfs = np.column_stack((X_test_a_U2Rfs, X_test_a_1))
            # df_test_U2R = processing.Standardization(df_test_U2R)
            X = np.array(df_U2R_fs)  # All rows, omit last column
            y = np.ravel(np.array(df_U2R)[:, -1:])  # All rows, only the last column
            # print(Counter(y))
            # smote
            sm = SMOTE(random_state=42)
            X_train_U2R, y_train_U2R = sm.fit_resample(X, y.astype('int'))
            # msmote
            # U2R_df = df[[each == 1 for each in df['label']]]
            # #for m in range(10000,15000,100):
            # msmote = ms.MSmote(N=m)
            # samples = msmote.fit(U2R_df)
            # df = np.append(pd.DataFrame(df), samples, axis=0)
            # df = pd.DataFrame(df)
            # X_train = np.array(df)[:, :-1]  # All rows, omit last column
            # y_train = np.ravel(np.array(df)[:, -1:])  # All rows, only the last column
            # normal
            # X_train_U2R = X
            # y_train_U2R = y
            # print(Counter(y))
            X_train_U2R = std.fit_transform(X_train_U2R)
            X_test_U2R = np.array(df_test_U2R_fs)  # All rows, omit last column
            y_test_U2R = np.ravel(np.array(df_test_U2R)[:, -1:])  # All rows, only the last column
            # print(Counter(y_test_U2R))

            # for i in range(19,30):
            rfc = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=18, )
            rfc.fit(X_train_U2R, y_train_U2R.astype('int'))
            y_pred_U2R = rfc.score(X_test_U2R, y_test_U2R.astype('int'))
            y_pred_t_U2R = rfc.fit(X_train_U2R, y_train_U2R.astype('int')).predict(X_test_U2R)
            # print("%d:%.10f"%(i,y_pred_U2R))
            # print(Counter(y_pred_t_U2R))
            # print(y_pred_t)
            X_train_NN = np.column_stack((X_train_1, rfc.predict_proba(X_test_U2R)))
            X_test_a_U2Rfs = std.fit_transform(X_test_a_U2Rfs)
            X_test_NN = np.column_stack((X_test_NN,rfc.predict_proba(X_test_a_U2Rfs)))
        else:
            X = np.array(df_U2R)[:, :-1]  # All rows, omit last column
            y = np.ravel(np.array(df_U2R)[:, -1:])  # All rows, only the last column
            # print(Counter(y))
            # smote
            sm = SMOTE(random_state=42)
            X_train_U2R, y_train_U2R = sm.fit_resample(X, y.astype('int'))
            # msmote
            # R2L_df =pd.DataFrame(df_R2L,columns=col_names)
            # R2L_df = R2L_df[[each == 1 for each in R2L_df['label']]]
            # #for m in range(10000,15000,100):
            # msmote = ms.MSmote(N=200)
            # samples = msmote.fit(R2L_df)
            # df_R2L = pd.DataFrame(np.append(pd.DataFrame(df_R2L), samples, axis=0))
            # X_train_R2L = np.array(df_R2L)[:, :-1]  # All rows, omit last column
            # y_train_R2L = np.ravel(np.array(df_R2L)[:, -1:])  # All rows, only the last column
            # normal
            # X_train_R2L = X
            # y_train_R2L = y
            # print(Counter(y))
            X_train_U2R = std.fit_transform(X_train_U2R)
            X_test_U2R = np.array(df_test_U2R)[:, :-1]  # All rows, omit last column
            y_test_U2R = np.ravel(np.array(df_test_U2R)[:, -1:])  # All rows, only the last column
            # print(Counter(y_test_R2L))

            # for i in range(1,20):
            rfc = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, )
            rfc.fit(X_train_U2R, y_train_U2R.astype('int'))
            y_pred_U2R = rfc.score(X_test_U2R, y_test_U2R.astype('int'))
            y_pred_t_U2R = rfc.fit(X_train_U2R, y_train_U2R.astype('int')).predict(X_test_U2R)
            # print("%d:%.10f"%(i,y_pred_U2R))
            # print(Counter(y_test_U2R))
            # print(Counter(y_pred_t_U2R))
                # print(y_pred_t)
            X_train_NN = np.column_stack((X_train_1, rfc.predict_proba(X_test_U2R)))
            X_test_a_U2R = std.fit_transform(X_test_b)
            X_test_NN = np.column_stack((X_test_NN, rfc.predict_proba(X_test_a_U2R)))

    # todo NN
        df_testing = pd.DataFrame(np.column_stack((X_test, y_test)), columns=col_names)
        df_train_NN = processing.changetag(df_testing)
        y_train_NN = np.ravel(np.array(df_train_NN)[:, -1:]) # All rows, omit last column
        # print(Counter(y_train_NN))
        X_train_NN, y_train_NN = sm.fit_resample(X_train_NN, y_train_NN.astype('int'))
        # print(Counter(y_train_NN))
        y = np.ravel(np.array(df_test)[:, -1:])  # All rows, only the last column
        print(Counter(y))

        # model = SVC(kernel='rbf', probability=True)
        # model.fit(X_train_NN,y_train_NN.astype('int'))
        # y_pred_NN = model.score(X_test_NN,y.astype('int'))
        # svc_model = SVC(kernel='rbf')
        # param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001,0.0001]}  # param_grid:我们要调参数的列表(带有参数名称作为键的字典)，此处共有14种超参数的组合来进行网格搜索，进而选择一个拟合分数最好的超平面系数。
        # grid_search = GridSearchCV(svc_model, param_grid, n_jobs=-1,verbose=1)  # n_jobs:并行数，int类型。(-1：跟CPU核数一致；1:默认值)；verbose:日志冗长度。默认为0：不输出训练过程；1：偶尔输出；>1：对每个子模型都输出。
        # grid_search.fit(X_train_NN,y_train_NN.astype('int'))  # 训练，默认使用5折交叉验证
        # s = grid_search.score
        # best_parameters = grid_search.best_estimator_.get_params()  # 获取最佳模型中的最佳参数
        # print("cv results are" % grid_search.best_params_, grid_search.cv_results_)  # grid_search.cv_results_:给出不同参数情况下的评价结果。
        # print("best parameters are" % grid_search.best_params_,grid_search.best_params_)  # grid_search.best_params_:已取得最佳结果的参数的组合；
        # print("best score are" % grid_search.best_params_, grid_search.best_score_)  # grid_search.best_score_:优化过程期间观察到的最好的评分。
        # # for para, val in list(best_parameters.items()):
        # #     print(para, val)
        svm_model = SVC(kernel='rbf', C=1000, gamma=0.001)  # 最佳模型
        # svm_model.fit(X_train_NN, y_train_NN.astype('int'))
        y_pred_t = svm_model.fit(X_train_NN, y_train_NN.astype('int')).predict(X_test_NN)
        print(Counter(y_pred_t))
        # print(metrics.precision_score(y.astype('int'), y_pred_t, labels=[5], average='macro'))
        # print(metrics.precision_score(y.astype('int'), y_pred_t, labels=[4], average='macro'))
        # print(metrics.precision_score(y.astype('int'), y_pred_t, labels=[3], average='macro'))
        # print(metrics.f1_score(y.astype('int'), y_pred_t, average='macro'))
        y_pred_NN = svm_model.score(X_test_NN,y.astype('int'))
        #
        # print(y_pred_NN)
        print ("%.10f:%.10f"%(y_pred_NN,metrics.f1_score(y.astype('int'), y_pred_t, average='macro')))
        print(metrics.classification_report(y, y_pred_t))
        cfm = confusion_matrix(y,y_pred_t)
        print(cfm)
        Ve = ["Normal","Dos","Probe","R2L","U2R"]
        heatmap(cfm,Ve,Ve)




# a =1