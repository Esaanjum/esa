import pandas as pd


def makePrediction(df):
    def change_into_datetime(col):
        df[col] = pd.to_datetime(df[col],errors='coerce')

    def extract_hour(df, col):
        df[col + '_hour'] = df[col].dt.hour

    def extract_minute(df, col):
        df[col + '_minute'] = df[col].dt.minute

    def drop_column(df, col):
        df.drop(col, axis=1, inplace=True)

    def extract_duration_hours(x):
        return x.split(' ')[0][0:-1]

    def extract_duration_mins(x):
        return x.split(' ')[1][0:-1]

    # Get original dataset and unpickle it
    unpickled_df = pd.read_pickle("df.pkl")
    # The input
    df['Price'] = 0
    # create dataframe of the input
    df = pd.DataFrame(df, index=[0])
    # Append the input to dataset
    df = unpickled_df.append(df)
    # # Converting Data Types
    for i in ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']:
        change_into_datetime(i)
    # # Processing datetime columns
    # # store day/month from  Date_of_Journey
    df['journey_month'] = df['Date_of_Journey'].dt.month
    df['journey_day'] = df['Date_of_Journey'].dt.day
    drop_column(df, 'Date_of_Journey')
    # store hour/minutes from Dep_Time and Arrival_Time
    extract_hour(df, 'Dep_Time')
    extract_minute(df, 'Dep_Time')
    drop_column(df, 'Dep_Time')
    extract_hour(df, 'Arrival_Time')
    extract_minute(df, 'Arrival_Time')
    drop_column(df, 'Arrival_Time')
    # splitting duration and adding '0m' wherever there is only hour present
    duration = list(df['Duration'])
    for i in range(len(duration)):
        if len(duration[i].split(' ')) == 2:
            pass
        else:
            if 'h' in duration[i]:
                duration[i] = duration[i] + ' 0m'
            else:
                duration[i] = '0h ' + duration[i]
    df['Duration'] = duration
    # store duration hours and minutes
    df['Duration_hours'] = df['Duration'].apply(extract_duration_hours)
    df['Duration_mins'] = df['Duration'].apply(extract_duration_mins)
    df['Duration_hours'] = df['Duration_hours'].astype(int)
    df['Duration_mins'] = df['Duration_mins'].astype(int)
    drop_column(df, 'Duration')

    # # storing names of columns with categorical data & continous data separately
    cat_col = [col for col in df.columns if df[col].dtype == 'O']
    cont_col = [col for col in df.columns if df[col].dtype != 'O']

    # Convert Categorical Data into Numerical using One-hot Encoding
    categorical = df[cat_col]
    Airline = pd.get_dummies(categorical['Airline'], drop_first=True)
    Source = pd.get_dummies(categorical['Source'], drop_first=True)
    Destination = pd.get_dummies(categorical['Destination'], drop_first=True)

    categorical["Route_1"] = categorical["Route"].str.split('→').str[0]
    categorical["Route_2"] = categorical["Route"].str.split('→').str[1]
    categorical["Route_3"] = categorical["Route"].str.split('→').str[2]
    categorical["Route_4"] = categorical["Route"].str.split('→').str[3]
    categorical["Route_5"] = categorical["Route"].str.split('→').str[4]
    categorical["Route_6"] = categorical["Route"].str.split('→').str[5]
    drop_column(categorical, 'Route')
    for i in ['Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5', 'Route_6']:
        categorical[i].fillna('None', inplace=True)

    # Label Encoding Columns that have a lot of categorical/alphabetical data
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    for i in ['Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5', 'Route_6']:
        categorical[i] = encoder.fit_transform(categorical[i])
    drop_column(categorical, 'Additional_Info')

    # Converting Total Stops Column data manually i.e. without Label Encoder
    dict = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    categorical['Total_Stops'] = categorical['Total_Stops'].map(dict)

    # Concatenating Data to dataframe
    data_train = pd.concat([categorical, Airline, Source, Destination, df[cont_col]], axis=1)
    # Dropping Columns that have categorical/alphabetical data and is no longer needed
    drop_column(data_train, 'Airline')
    drop_column(data_train, 'Source')
    drop_column(data_train, 'Destination')
    pd.set_option('display.max_columns', 35)

    X = data_train.drop('Price', axis=1)
    # get the last row i.e. input
    inp = X.iloc[-1:]

    import pickle

    # load the model from disk
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    try:
        prediction = "{:.2f}".format(float(loaded_model.predict(inp)))
    except:
        pass
    prediction = ''.join(map(str, prediction))
    return prediction

# makePrediction(df = {'Airline': 'IndiGo', 'Date_of_Journey': '24/03/2019', 'Source': 'Banglore', 'Destination': 'New Delhi',
#           'Route': 'BLR → DEL', 'Dep_Time': '22:20', 'Arrival_Time': '01:10 22 Mar', 'Duration': '2h 50m',
#           'Total_Stops': 'non-stop', 'Additional_Info': 'No info'})