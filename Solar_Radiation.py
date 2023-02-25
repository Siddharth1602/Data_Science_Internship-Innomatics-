import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


st.set_page_config(page_title= "Solar Irradiance prediction",
                   page_icon= ":bar_chart:",layout= "wide")
st.title(":bar_chart: Solar Irradiance Prediction")
st.subheader("Created by Siddharth Under the supervision of :red[Innomatics] Research Labs.")



Solar_prediction = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Internship/Project_1/SolarPrediction.csv")
st.sidebar.title("Solar Prediction")
option = st.sidebar.selectbox(
    'Select Your Chart',
    ('Temperature Graph', 'Pressure Graph'
     , 'Humidity Graph', 'Correlation Matrix'))

skewness = st.sidebar.selectbox(
    'Check the Skewness in Dataset',
    ('Temperature', 'Pressure'
     , 'Humidity'))

predictions = st.sidebar.selectbox(
   "Check the Errors Based on Linear Regression Model",
   ("Mean Absolute Error","Mean Squared Error","Root Mean Squared Error")
)

# Display a chart based on the selected option

if option == "Temperature Graph":
   chart_data = Solar_prediction["Temperature"]
   fig, ax = plt.subplots()
   ax.hist(chart_data, bins=20)
   st.pyplot(fig)
elif option == "Pressure Graph":
   chart_data2 = Solar_prediction["Pressure"]
   fig,ax = plt.subplots()
   ax.hist(chart_data2, bins = 20)
   st.pyplot(fig)
elif option == "Correlation Matrix":
   fig, ax = plt.subplots()
   corr_matrix = Solar_prediction.corr()
   sns.heatmap(corr_matrix)
   st.pyplot(fig)
else:
   chart_data3 = Solar_prediction["Humidity"]
   fig,ax = plt.subplots()
   fig,ax = plt.subplots()
   ax.hist(chart_data3,bins =20)
   st.pyplot(fig)
st.markdown("---")
# Display a chart based on the skewness option

if skewness == 'Temperature':
   fig, ax = plt.subplots()
   sns.distplot(Solar_prediction["Temperature"])
   st.pyplot(fig)
elif skewness == "Pressure":
   fig, ax = plt.subplots()
   sns.distplot(Solar_prediction["Pressure"])
   st.pyplot(fig)
else:
   fig, ax = plt.subplots()
   sns.distplot(Solar_prediction["Humidity"])
   st.pyplot(fig)



## Creating Predictions:

Solar_prediction['TimeSunRise'] = pd.to_datetime(Solar_prediction['TimeSunRise'])
Solar_prediction['TimeSunSet'] = pd.to_datetime(Solar_prediction['TimeSunSet'])

# Convert the TimeSunRise and TimeSunSet columns to the number of seconds since midnight
Solar_prediction['TimeSunRise'] = Solar_prediction['TimeSunRise'].dt.hour * 3600 + Solar_prediction['TimeSunRise'].dt.minute * 60 + Solar_prediction['TimeSunRise'].dt.second
Solar_prediction['TimeSunSet'] = Solar_prediction['TimeSunSet'].dt.hour * 3600 + Solar_prediction['TimeSunSet'].dt.minute * 60 + Solar_prediction['TimeSunSet'].dt.second

X = Solar_prediction[['Temperature', 'TimeSunSet', 'TimeSunRise']]
y = Solar_prediction[['Humidity', 'TimeSunSet', 'TimeSunRise']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

Mean_Absolute_Error = metrics.mean_absolute_error(y_test, y_pred)
Mean_Squared_Error = metrics.mean_squared_error(y_test, y_pred)
Root_Mean_Error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#Displaying Predictions:
st.markdown('---')
if predictions == "Mean Absolute Error":
   st.write("**The mean absolute error is:**",Mean_Absolute_Error)
elif predictions == "Mean Squared Error":
   st.write("**The mean squared error is:**",Mean_Squared_Error)
else:
   st.write("**The Root Mean Square Error is:**",Root_Mean_Error)

st.markdown('---')   
st.header("The predicted values on the basis of Linear Regression Model")

df1= pd.DataFrame(y_pred)
df1 = df1.rename(columns ={0:'Temperature',1:'TimeSunSet',2:'TimeSunRise'})
st.dataframe(df1)







##Creating the Download Button
@st.cache_data
def convert_df(Solar_prediction):
   return Solar_prediction.to_csv(index=False).encode('utf-8')


csv = convert_df(Solar_prediction)

st.download_button(
   "Download the DataSet",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)
