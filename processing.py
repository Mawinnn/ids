import pandas as pd



def changetag_Normal(df):
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    labeldf = df['label']
    newlabeldf = labeldf.replace(
        {'normal': 1,
         'neptune': 0, 'back': 0, 'land': 0, 'pod': 0, 'smurf': 0, 'teardrop': 0, 'mailbomb': 0,'apache2': 0,'processtable': 0, 'udpstorm': 0, 'worm': 0,
         'ipsweep': 0, 'nmap': 0, 'portsweep': 0, 'satan': 0, 'mscan': 0, 'saint': 0,
         'ftp_write': 0, 'guess_passwd': 0, 'imap': 0, 'multihop': 0, 'phf': 0, 'spy': 0, 'warezclient': 0,'warezmaster': 0, 'sendmail': 0, 'named': 0, 'snmpgetattack': 0, 'snmpguess': 0, 'xlock': 0, 'xsnoop': 0,'httptunnel': 0,
         'buffer_overflow': 0, 'loadmodule': 0, 'perl': 0, 'rootkit': 0, 'ps': 0, 'sqlattack': 0, 'xterm': 0})
    df['label'] = newlabeldf
    return df

def changetag_Dos(df):
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
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
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    labeldf = df['label']
    newlabeldf = labeldf.replace(
        {'normal': 0,
         'neptune': 0, 'back': 0, 'land': 0, 'pod': 0, 'smurf': 0, 'teardrop': 0, 'mailbomb': 0,'apache2': 0,'processtable': 0, 'udpstorm': 0, 'worm': 0,
         'ipsweep': 1, 'nmap': 1, 'portsweep': 1, 'satan': 1, 'mscan': 1, 'saint': 1,
         'ftp_write': 0, 'guess_passwd': 0, 'imap': 0, 'multihop': 0, 'phf': 0, 'spy': 0, 'warezclient': 0,'warezmaster': 0, 'sendmail': 0, 'named': 0, 'snmpgetattack': 0, 'snmpguess': 0, 'xlock': 0, 'xsnoop': 0,'httptunnel': 0,
         'buffer_overflow': 0, 'loadmodule': 0, 'perl': 0, 'rootkit': 0, 'ps': 0, 'sqlattack': 0, 'xterm': 0})
    df['label'] = newlabeldf
    return df

def changetag_R2L(df):
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
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

def changetag_U2R(df):
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
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
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    labeldf = df['label']
    newlabeldf = labeldf.replace(
        {'normal': 1,
         'neptune': 2, 'back': 2, 'land': 2, 'pod': 2, 'smurf': 2, 'teardrop': 2, 'mailbomb': 2,'apache2': 2,'processtable': 2, 'udpstorm': 2, 'worm': 2,
         'ipsweep': 3, 'nmap': 3, 'portsweep': 3, 'satan': 3, 'mscan': 3, 'saint': 3,
         'ftp_write': 4, 'guess_passwd': 4, 'imap': 4, 'multihop': 4, 'phf': 4, 'spy': 4, 'warezclient': 4,'warezmaster': 4, 'sendmail': 4, 'named': 4, 'snmpgetattack': 4, 'snmpguess': 4, 'xlock': 4, 'xsnoop': 4,'httptunnel': 4,
         'buffer_overflow': 5, 'loadmodule': 5, 'perl': 5, 'rootkit': 5, 'ps': 5, 'sqlattack': 5, 'xterm': 5})
    df['label'] = newlabeldf
    return df