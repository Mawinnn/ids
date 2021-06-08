from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pandas as pd

def change(df,df_test):
    for col_name in df.columns:
        if df[col_name].dtypes == 'object':
            unique_cat = len(df[col_name].unique())
            #print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))
    for col_name in df_test.columns:
        if df_test[col_name].dtypes == 'object':
            unique_cat = len(df_test[col_name].unique())
            # print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))
    categorical_columns = ['protocol_type', 'service', 'flag']
    # Get the categorical values into a 2D numpy array
    df_categorical_values = df[categorical_columns]
    testdf_categorical_values = df_test[categorical_columns]

    unique_protocol = sorted(df.protocol_type.unique())
    string1 = 'Protocol_type_'
    unique_protocol2 = [string1 + x for x in unique_protocol]
    # service
    unique_service = sorted(df.service.unique())
    string2 = 'service_'
    unique_service2 = [string2 + x for x in unique_service]
    # flag
    unique_flag = sorted(df.flag.unique())
    string3 = 'flag_'
    unique_flag2 = [string3 + x for x in unique_flag]
    # put together
    dumcols = unique_protocol2 + unique_service2 + unique_flag2
    print(dumcols)

    # do same for test set
    unique_service_test = sorted(df_test.service.unique())
    unique_service2_test = [string2 + x for x in unique_service_test]
    testdumcols = unique_protocol2 + unique_service2_test + unique_flag2

    df_categorical_values_enc = df_categorical_values.apply(LabelEncoder().fit_transform)
    print(df_categorical_values_enc.head())
    # test set
    testdf_categorical_values_enc = testdf_categorical_values.apply(LabelEncoder().fit_transform)

    enc = OneHotEncoder()
    df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
    df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(), columns=dumcols)
    # test set
    testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
    testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(), columns=testdumcols)

    trainservice = df['service'].tolist()
    testservice = df_test['service'].tolist()
    difference = list(set(trainservice) - set(testservice))
    string = 'service_'
    difference = [string + x for x in difference]

    for col in difference:
        testdf_cat_data[col] = 0
    newdf = df.join(df_cat_data)
    newdf.drop('flag', axis=1, inplace=True)
    newdf.drop('protocol_type', axis=1, inplace=True)
    newdf.drop('service', axis=1, inplace=True)
    # test data
    newdf_test = df_test.join(testdf_cat_data)
    newdf_test.drop('flag', axis=1, inplace=True)
    newdf_test.drop('protocol_type', axis=1, inplace=True)
    newdf_test.drop('service', axis=1, inplace=True)
    newdf = pd.concat([newdf, newdf.pop('label')], 1)
    newdf_test = pd.concat([newdf_test, newdf_test.pop('label')], 1)
    return newdf,newdf_test


def changetag_R2L(df):
    labeldf = df['label']
    newlabeldf = labeldf.replace(
        {'normal': 0, 'neptune': 0, 'back': 0, 'land': 0, 'pod': 0, 'smurf': 0, 'teardrop': 0, 'mailbomb': 0,
         'apache2': 0,
         'processtable': 0, 'udpstorm': 0, 'worm': 0,
         'ipsweep': 0, 'nmap': 0, 'portsweep': 0, 'satan': 0, 'mscan': 0, 'saint': 0
            , 'ftp_write': 1, 'guess_passwd': 1, 'imap': 1, 'multihop': 1, 'phf': 1, 'spy': 1, 'warezclient': 1,
         'warezmaster': 1, 'sendmail': 1, 'named': 1, 'snmpgetattack': 1, 'snmpguess': 1, 'xlock': 1, 'xsnoop': 1,
         'httptunnel': 1,
         'buffer_overflow': 0, 'loadmodule': 0, 'perl': 0, 'rootkit': 0, 'ps': 0, 'sqlattack': 0, 'xterm': 0})
    df['label'] = newlabeldf
    return df

def Standardization(KDD):
    KDD = (KDD - KDD.min()) / (KDD.max() - KDD.min())
    KDD = KDD.fillna(0)
    return KDD

def changetag_Normal(df):
    labeldf = df['label']
    newlabeldf = labeldf.replace(
        {'normal': 0,
         'neptune': 1, 'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1, 'mailbomb': 1,'apache2': 1,'processtable': 1, 'udpstorm': 1, 'worm': 1,
         'ipsweep': 1, 'nmap': 1, 'portsweep': 1, 'satan': 1, 'mscan': 1, 'saint': 1,
         'ftp_write': 1, 'guess_passwd': 1, 'imap': 1, 'multihop': 1, 'phf': 1, 'spy': 1, 'warezclient': 1,'warezmaster': 1, 'sendmail': 1, 'named': 1, 'snmpgetattack': 1, 'snmpguess': 1, 'xlock': 1, 'xsnoop': 1,'httptunnel': 1,
         'buffer_overflow': 1, 'loadmodule': 1, 'perl': 1, 'rootkit': 1, 'ps': 1, 'sqlattack': 1, 'xterm': 1})
    df['label'] = newlabeldf
    return df

def changetag_Dos(df):
    labeldf = df['label']
    newlabeldf = labeldf.replace(
        {'normal': 0,
         'neptune': 1, 'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1, 'mailbomb': 1,'apache2': 1,'processtable': 1, 'udpstorm': 1, 'worm': 1,
         'ipsweep': 0, 'nmap': 0, 'portsweep': 0, 'satan': 0, 'mscan': 0, 'saint': 0,
         'ftp_write': 0, 'guess_passwd': 0, 'imap': 0, 'multihop': 0, 'phf': 0, 'spy': 0, 'warezclient': 0,'warezmaster': 0, 'sendmail': 0, 'named': 0, 'snmpgetattack': 0, 'snmpguess': 0, 'xlock': 0, 'xsnoop': 0,'httptunnel': 0,
         'buffer_overflow': 0, 'loadmodule': 0, 'perl': 0, 'rootkit': 0, 'ps': 0, 'sqlattack': 0, 'xterm': 0})
    df['label'] = newlabeldf
    return df

def changetag_Probe(df):
    labeldf = df['label']
    newlabeldf = labeldf.replace(
        {'normal': 0,
         'neptune': 0, 'back': 0, 'land': 0, 'pod': 0, 'smurf': 0, 'teardrop': 0, 'mailbomb': 0,'apache2': 0,'processtable': 0, 'udpstorm': 0, 'worm': 0,
         'ipsweep': 1, 'nmap': 1, 'portsweep': 1, 'satan': 1, 'mscan': 1, 'saint': 1,
         'ftp_write': 0, 'guess_passwd': 0, 'imap': 0, 'multihop': 0, 'phf': 0, 'spy': 0, 'warezclient': 0,'warezmaster': 0, 'sendmail': 0, 'named': 0, 'snmpgetattack': 0, 'snmpguess': 0, 'xlock': 0, 'xsnoop': 0,'httptunnel': 0,
         'buffer_overflow': 0, 'loadmodule': 0, 'perl': 0, 'rootkit': 0, 'ps': 0, 'sqlattack': 0, 'xterm': 0})
    df['label'] = newlabeldf
    return df

def changetag_U2R(df):
    labeldf = df['label']
    newlabeldf = labeldf.replace(
        {'normal': 0,
         'neptune': 0, 'back': 0, 'land': 0, 'pod': 0, 'smurf': 0, 'teardrop': 0, 'mailbomb': 0,'apache2': 0,'processtable': 0, 'udpstorm': 0, 'worm': 0,
         'ipsweep': 0, 'nmap': 0, 'portsweep': 0, 'satan': 0, 'mscan': 0, 'saint': 0,
         'ftp_write': 0, 'guess_passwd': 0, 'imap': 0, 'multihop': 0, 'phf': 0, 'spy': 0, 'warezclient': 0,'warezmaster': 0, 'sendmail': 0, 'named': 0, 'snmpgetattack': 0, 'snmpguess': 0, 'xlock': 0, 'xsnoop': 0,'httptunnel': 0,
         'buffer_overflow': 1, 'loadmodule': 1, 'perl': 1, 'rootkit': 1, 'ps': 1, 'sqlattack': 1, 'xterm': 1})
    df['label'] = newlabeldf
    return df
def changetag(df):
    labeldf = df['label']
    newlabeldf = labeldf.replace(
        {'normal': 1,
         'neptune': 2, 'back': 2, 'land': 2, 'pod': 2, 'smurf': 2, 'teardrop': 2, 'mailbomb': 2,'apache2': 2,'processtable': 2, 'udpstorm': 2, 'worm': 2,
         'ipsweep': 3, 'nmap': 3, 'portsweep': 3, 'satan': 3, 'mscan': 3, 'saint': 3,
         'ftp_write': 4, 'guess_passwd': 4, 'imap': 4, 'multihop': 4, 'phf': 4, 'spy': 4, 'warezclient': 4,'warezmaster': 4, 'sendmail': 4, 'named': 4, 'snmpgetattack': 4, 'snmpguess': 4, 'xlock': 4, 'xsnoop': 4,'httptunnel': 4,
         'buffer_overflow': 5, 'loadmodule': 5, 'perl': 5, 'rootkit': 5, 'ps': 5, 'sqlattack': 5, 'xterm': 5})
    df['label'] = newlabeldf
    return df
