# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# %%
# Read the input file into a pandas DataFrame
df = pd.read_csv('input-data.csv')
df
# %%
# Drop rows with NaN values in specified columns
columns_to_check = ['start date', 'total attendance', 'search volume 1']
df.dropna(subset=columns_to_check, inplace=True)

# %%
# Convert the start date and end date columns to datetime objects
df['start date'] = pd.to_datetime(df['start date'], format='%d/%m/%Y')
df['end date'] = pd.to_datetime(df['end date'], format='%d/%m/%Y')

# %%
# Perform one-hot encoding on the 'gallery' column
df_encoded = pd.get_dummies(df, columns=['gallery'], prefix='gallery', drop_first=True)
df_encoded
# %%

# Split the data into input features (X) and target variable (y)
X = df_encoded[['gallery_Main', 'gallery_Sackler', 'gallery_GJW', 'number of days', 'search volume 1']]
y = df_encoded['total attendance']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate error metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)

# %%
# Calculate percentage difference
percentage_diff = (abs(y_pred - y_test) / y_test) * 100

# Create a DataFrame to store the results
percent_diff_df = pd.DataFrame({'Actual Total Attendance': y_test,
                           'Predicted Total Attendance': y_pred,
                           'Percentage Difference': percentage_diff})

percent_diff_df

# Calculate mean absolute percentage error
mape = mean_absolute_percentage_error(y_test, y_pred)

print('Mean Absolute Percentage Error (MAPE):', mape)
print('Percentage Difference:')
print(percent_diff_df)

# %%
# Example usage for a new proposed exhibition
new_gallery = 'Art Gallery A'
new_start_date = pd.to_datetime('2023-06-01')
new_end_date = pd.to_datetime('2023-06-15')
new_number_of_days = (new_end_date - new_start_date).days
new_online_search_volume = 1000

# Create a DataFrame for the new exhibition
new_exhibition = pd.DataFrame([[0, 1, 0, 90, 12000]],
                              columns=['gallery_Main', 'gallery_Sackler', 'gallery_GJW', 'number of days', 'search volume 1'])

# Make prediction for the new exhibition
new_attendance_prediction = model.predict(new_exhibition)
print('Predicted Total Attendance for the New Exhibition:', new_attendance_prediction[0])

# %%
