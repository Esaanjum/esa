import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

def plot(df,col):
    fig,(ax1,ax2) = plt.subplots(2,1)
    sns.histplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)


df = pd.read_excel('Data_Train.xlsx')
df.to_pickle("df.pkl")
print("----------Data Set----------")
print(df.head())

# Removing Null Values
print("Null Values Before:\n", df.isna().sum())
df.dropna(inplace=True)  # inplace is used to update dataframe
print("Null Values After:\n", df.isna().sum())

# Converting Data Types
print("\nData Types Before:")
print(df.dtypes)
for i in ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']:
    change_into_datetime(i)
print("Data Types After:")
print(df.dtypes)

# Plot Total Stops Graph w.r.t Price
plt.figure(figsize=(15,5))
sns.boxplot(x='Total_Stops',y='Price',data=df.sort_values('Price',ascending=False))
plt.savefig('D:/Data Science/Project/static/total_stops.png')

# Plot Graph of Price in relation to Date
df1 = pd.DataFrame()
df1['Date_of_Journey']= df['Date_of_Journey'].dt.date
df1['Price'] = df['Price']
pa=sns.catplot(x='Date_of_Journey', y='Price', data=df1.sort_values('Date_of_Journey',ascending=False),kind="point",height = 5, aspect = 3)
plt.xlabel('Travelling Date')
plt.subplots_adjust(bottom=0.25,left=0.2)
plt.xticks(rotation=70)
# plt.figure(figsize = (15,5))
# dfPlot = df[['Date_of_Journey','Price']]
# dfPlot.set_index('Date_of_Journey',inplace=True)
# dfPlot['Price'].plot()
plt.savefig('D:/Data Science/Project/static/price_time_series.png')
# Processing datetime columns
# store day/month from  Date_of_Journey
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
# splitting duration and adding '0m' or '0h' wherever there is only hour present
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

# storing names of columns with categorical data & continous data separately
cat_col = [col for col in df.columns if df[col].dtype == 'O']
cont_col = [col for col in df.columns if df[col].dtype != 'O']
print("\nCategorical Data Columns: ", cat_col)
print("Continous Data Colunms: ", cont_col)
print("\n")
print("------------------After Processing Datetime Columns------------------")
print(df.head())
print(df.dtypes)

# Plot Graph of Price w.r.t Journey Month
plt.figure(figsize=(15,5))
sns.boxplot(x='journey_month',y='Price',data=df.sort_values('Price',ascending=False))
plt.savefig('D:/Data Science/Project/static/journey_month.png')
# Plot Graph of Price w.r.t Journey Day
plt.figure(figsize=(15,5))
sns.boxplot(x='journey_day',y='Price',data=df.sort_values('Price',ascending=False))
plt.savefig('D:/Data Science/Project/static/journey_day.png')

# Convert Categorical Data into Numerical using One-hot Encoding
categorical = df[cat_col]
Airline = pd.get_dummies(categorical['Airline'], drop_first=True)
Source = pd.get_dummies(categorical['Source'], drop_first=True)
Destination = pd.get_dummies(categorical['Destination'], drop_first=True)
print("----------------Example of One-hot Encoding of Destination Column----------------")
print(Destination.head())

categorical["Route_1"] = categorical["Route"].str.split('→').str[0]
categorical["Route_2"] = categorical["Route"].str.split('→').str[1]
categorical["Route_3"] = categorical["Route"].str.split('→').str[2]
categorical["Route_4"] = categorical["Route"].str.split('→').str[3]
categorical["Route_5"] = categorical["Route"].str.split('→').str[4]
categorical["Route_6"] = categorical["Route"].str.split('→').str[5]

drop_column(categorical, 'Route')
print("Columns with Null Values to fill:", categorical.isnull().sum())
for i in ['Route_3', 'Route_4', 'Route_5', 'Route_6']:
    categorical[i].fillna('None', inplace=True)
for i in categorical.columns:
    print('{} has total {} categories'.format(i, len(categorical[i].value_counts())))

# Label Encoding Columns that have a lot of categorical/alphabetical data
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for i in ['Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5', 'Route_6']:
    categorical[i] = encoder.fit_transform(categorical[i])
drop_column(categorical, 'Additional_Info')

# Converting Total Stops Column data manually i.e. without Label Encoder
print("Total Stops column values to convert into Numerical Data:\n")
print(categorical['Total_Stops'].unique())
dict = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
categorical['Total_Stops'] = categorical['Total_Stops'].map(dict)

# Concatenating Data to dataframe
data_train = pd.concat([categorical, Airline, Source, Destination, df[cont_col]], axis=1)
# Dropping Columns that have categorical/alphabetical data and is no longer needed
drop_column(data_train, 'Airline')
drop_column(data_train, 'Source')
drop_column(data_train, 'Destination')
pd.set_option('display.max_columns', 35)
print(data_train.head())


# Plot Graph of Outliers
plt.figure(figsize=(30,20))
plot(data_train,'Price')
plt.savefig('D:/Data Science/Project/static/outliers.png')

# Handling Outliers
data_train['Price'] = np.where(data_train['Price'] >= 40000, data_train['Price'].median(), data_train['Price'])
# Plot Graph of fixed Outliers
plt.figure(figsize = (30,20))
plot_outlier = plot(data_train,'Price')
plt.savefig('D:/Data Science/Project/static/fixed_outliers.png')
data_train.dropna(inplace=True)
X = data_train.drop('Price', axis=1)
Y = data_train['Price']

# print("Columns with Null Values to fill:", X.isnull().sum())
# for i in ['Arrival_Time_hour', 'Arrival_Time_minute']:
#     X[i] = X[i].fillna(method ='ffill', inplace=False)



# Feature Selection
from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, Y)
# Find Feature Importance
imp = pd.DataFrame(selection.feature_importances_, index=X.columns)
imp.columns = ['importance']
imp.sort_values(by='importance', ascending=False)
print("Feature Importance:")
print(imp)

# Plot feature importance Graph
plt.figure(figsize = (12,8))
feat_importances = pd.Series(imp["importance"], index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.savefig('D:/Data Science/Project/static/feature_importance.png')

# Splitting dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

from sklearn import metrics


# Generating Graphs and Dumping model using pickle so that we can re-use
def predict(ml_model):
    model = ml_model.fit(X_train, y_train)

    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    print('Training score : {}'.format(model.score(X_train, y_train)))
    y_prediction = model.predict(X_test)
    print('predictions are: \n {}'.format(y_prediction))
    print('\n')
    r2_score = metrics.r2_score(y_test, y_prediction)
    print('r2 score: {}'.format(r2_score))
    print('MAE:', metrics.mean_absolute_error(y_test, y_prediction))
    print('MSE:', metrics.mean_squared_error(y_test, y_prediction))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))
    # Plot Random Forest Accuracy Graph
    sns_plot = sns.displot(y_test - y_prediction)
    sns_plot.set(xlabel='Deviation in Price', ylabel='No. of Predictions')
    sns_plot.savefig("D:/Data Science/Project/static/random_forest_accuracy.png")


# Predicting
from sklearn.ensemble import RandomForestRegressor

print(predict(RandomForestRegressor()))

