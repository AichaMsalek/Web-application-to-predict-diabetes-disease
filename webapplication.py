import numpy as np
import pickle
import streamlit as st
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the saved model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))
data = pd.read_csv('diabetes.csv')

st.sidebar.markdown("<h1 style='text-align: center;'>Choose Between :</h1>", unsafe_allow_html=True)
option = st.sidebar.selectbox('', ['Data Visualization', 'Diabetes Prediction'])

if option == 'Diabetes Prediction':
  def diabetes_prediction(input_data):
      # changing the input_data to numpy array
      input_data_as_numpy_array = np.asarray(input_data)

      # reshape the array as we are predicting for one instance
      input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
      prediction = loaded_model.predict(input_data_reshaped)
      print(prediction)

      if (prediction[0] == 0):
        return 'The person is not diabetic'
      else:
        return 'The person is diabetic'
    
  def main():
        
      st.title('Diabetes Prediction Web App 💉')
      st.write("Hello! Please enter the values ​​below to find out if you are diabetic or not")

      # getting the input data from the user 
      Pregnancies = st.text_input('Number of Pregnancies')
      Glucose = st.text_input('Glucose Level')
      BloodPressure = st.text_input('Blood Pressure value')
      SkinThickness = st.text_input('Skin Thickness value')
      Insulin = st.text_input('Insulin Level')
      BMI = st.text_input('BMI value')
      DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
      Age = st.text_input('Age of the Person')
      
      # code for prediction
      diagnosis = ''
      
      # creating a button for prediction
      if st.button('Diabetes Test Result'):
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            if 'not diabetic' in diagnosis.lower():
                st.success(diagnosis)
            else:
                st.error(diagnosis)
   
  if __name__ == '__main__':
    main()

elif option == 'Data Visualization':

  st.markdown("<h1 style='text-align: center; color:thistle;'>Data Visualization</h1>", unsafe_allow_html=True)

  if st.checkbox('Display Raw Data'):
      st.write(data)

  # Distribution of values
  st.markdown("<h3 style='font-size: 23px;text-align: center; color: lightslategray;'>Distribution of Values</h3>", unsafe_allow_html=True)
  st.write(data.describe())
  sns.set_palette("PuBuGn") 

  # Histogram
  st.markdown("<h3 style='font-size: 23px;text-align: center; color: lightslategray;'>Values Histogram</h3>", unsafe_allow_html=True)
  selected_column = st.selectbox('Select a column for the histogramm', data.columns)
  hist_fig, hist_ax = plt.subplots(figsize=(20, 8))
  hist_ax.hist(data[selected_column], bins=10, color='lavender')
  hist_ax.set_xlabel(selected_column)
  hist_ax.set_ylabel('Frequency')
  st.pyplot(hist_fig)


  # Create a dataframe for diabetes/non-diabetes
  diabetes_cases = data[data['Outcome'] == 1]
  non_diabetes_cases = data[data['Outcome'] == 0]
  num_diabetes_cases = len(diabetes_cases)
  num_non_diabetes_cases = len(non_diabetes_cases)
  # Créer une disposition en ligne
  col1, col2 = st.columns(2)

  with col1:
      st.markdown("<h3 style='font-size: 23px;text-align: center; color: lightslategray;'>Percentage of Diabetes vs Non-Diabetes</h3>", unsafe_allow_html=True)
      bar_fig, bar_ax = plt.subplots()
      sns.countplot(x='Outcome', data=data, ax=bar_ax)
      bar_ax.set_xlabel('Outcome')
      bar_ax.set_ylabel('Number of cases')
      st.pyplot(bar_fig)

  with col2:
      st.markdown("<h3 style='font-size: 23px;text-align: center; color: lightslategray;'>Percentage of Diabetes vs Non-Diabetes</h3>", unsafe_allow_html=True)
      labels = 'Diabetes', 'Non-diabetes'
      sizes = [num_diabetes_cases, num_non_diabetes_cases]
      fig, ax = plt.subplots()
      ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
      ax.axis('equal')  
      st.pyplot(fig)

  # Stacked Bar Chart
  st.markdown("<h3 style='font-size: 23px;text-align: center; color: lightslategray;'>Stacked Bar Chart for 'Age' and 'Outcome'</h3>", unsafe_allow_html=True)
  outcome_age = data.groupby(['Age', 'Outcome']).size().unstack()
  stacked_bar_fig, stacked_bar_ax = plt.subplots(figsize=(20, 8))
  outcome_age.plot(kind='bar', stacked=True, ax=stacked_bar_ax)
  stacked_bar_ax.set_xlabel('Age')
  stacked_bar_ax.set_ylabel('Number of cases')
  st.pyplot(stacked_bar_fig)

  # Scatter Plot
  st.markdown("<h3 style='font-size: 23px;text-align: center; color: lightslategray;'>Scatter Plot</h3>", unsafe_allow_html=True)
  x_column = st.selectbox('Select a column for X', data.columns)
  y_column = st.selectbox('Select a column for Y', data.columns)
  scatter_fig, scatter_ax = plt.subplots(figsize=(20, 8))
  scatter_ax.scatter(data[x_column], data[y_column], c=data['Outcome'])
  scatter_ax.set_xlabel(x_column)
  scatter_ax.set_ylabel(y_column)
  st.pyplot(scatter_fig)

  # Correlation Matrix
  st.markdown("<h3 style='font-size: 23px;text-align: center; color: lightslategray;'>Correlation Matrix</h3>", unsafe_allow_html=True)
  corr_matrix = data.corr()
  fig, ax = plt.subplots(figsize=(20, 7))
  dataplot = sns.heatmap(data=corr_matrix, annot=True, ax=ax,cmap='PuBuGn')
  st.pyplot(fig)










  






