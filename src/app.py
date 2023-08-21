#importing librabries

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import DMatrix  
import joblib
import traceback

# Load your trained XGBoost model from the JSON file
def load_xgb_model():
    dt_model = xgb.Booster()
    dt_model.load_model('assets/Ml_comp/xgb_model.json')
    return dt_model

    # Load the preprocessing components from the joblib files
def load_preprocessing_components():
    encoder = joblib.load('src/Asset/ML_Comp/encoder.joblib')
    scaler = joblib.load('src/Asset/ML_Comp/scaler.joblib')
    num_imputer = joblib.load('src/Asset/ML_Comp/numerical_imputer.joblib')
    cat_imputer = joblib.load('src/Asset/ML_Comp/categorical_imputer.joblib')
    return encoder, scaler, num_imputer, cat_imputer

# Load the preprocessing components
encoder, scaler, num_imputer, cat_imputer = load_preprocessing_components()

    
# Load your dataset
train_data = pd.read_csv("src/Asset/ML_Comp/train_final.csv", index_col=1)
train_data.rename(columns={'type_x': 'store_type'}, inplace=True)

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])  # Two pages

@st.cache_resource(experimental_allow_widgets=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value
        def get_value(val,my_dict):
            for key,value in my_dict.items():
                if val == key:
                    return value
# Function to load the dataset
@st.cache_resource()
def load_data(train_data):
    train_data["onpromotion"] = train_data["onpromotion"].apply(int)
    train_data["store_nbr"] = train_data["store_nbr"].apply(int)
    train_data["Sales_date"] = pd.to_datetime(train_data["Sales_date"]).dt.date
    train_data["year"] = pd.to_datetime(train_data['Sales_date']).dt.year
    return train_data

# Function to get date features from the inputs
def getDateFeatures(df, date):
    df['date'] = pd.to_datetime(df[date])
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    df['is_weekend'] = np.where(df['day_of_week'] > 4, 1, 0)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_end.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
    df = df.drop(columns="date")
    return df


if app_mode=='Home':
   st.title('Predictive Analytics for Grocery Sales Forecasting.')
   st.image('src/Asset/ML_Comp/img1.png')
   st.sidebar.header("This is a time series forecasting problem. The Favorita Grocery Sales Forecasting dataset is a fascinating collection of data that provides a great opportunity for analysis and prediction. We explore its various attributes, and analyze the sales patterns to build a robust sales forecasting model and answer some pertinent question on the dataset.") 
  
   page_mode = st.selectbox('Select your option',['Data Set Description','Data Exploratory Analysis','Results','You have not selected','Conclusion']) #four forms or pages
  
   if page_mode=='Data Set Description':
       st.header("Data Set Description")
       st.write("The dataset contains information about the sales of various products in different stores belonging to the Favorita chain of grocery stores in Ecuador.")
       st.write("The original train dataset has a total of 3,000,888 rows and 5 columns. The columns in the train dataset include the date, store number,")
       st.write("sales, family, and on promotion. The data in the dataset ranges from January 2013 to August 2017.")
       st.write("Additional files include supplementary information that may be useful in building the models.")
       st.title("File Descriptions and Data Field Information")
       st.write("The training data, comprising time series of features store_nbr, family, and onpromotion as well as the target sales.")
       st.write("store_nbr:  identifies the store at which the products are sold.")
       st.write("Family: identifies the type of product sold")
       st.write("Sales: gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units.")
       st.write("Onpromotion: gives the total number of items in a product family that were being promoted at a store at a given date.")
       st.markdown('Dataset') 
       st.write(train_data.head())
       
   elif page_mode=='Data Exploratory Analysis':
       st.header("Data Exploratory Analysis")
       st.write("Before we start building a sales forecasting model, we need to explore the data and analyze the patterns in the sales data. We start by importing the dataset and analyzing its different attributes.") 
       st.write("We can then use various visualization techniques to understand the trends in the data.")
       st.write("One interesting visualization is to plot the sales data against time. This gives us a clear idea of how sales have changed over the years. We can see that sales have been increasing gradually over time, with a few sharp peaks and drops.")
       st.image("src/Asset/ML_Comp/img3 (2).png")
       st.write ("From the chart above we can say that, sales have increased over time from 2013 to 2016. However,")
       st.write("there was a sharp decline in sales from 2016 to 2017.") 
       st.write("This can be as a result of earthquake that occurred on 16th April, 2016.")
                 
       st.image("src/Asset/ML_Comp/img2.jpg")       
       st.write("Here, we plotted distribution of each store type and the number of stores in that category. ")
       st.write("It was observed that store type D had the majority share, followed by type C with A, B and E in that order.")
       st.write("AGuayaquil and Quito are two cities that stand out in terms of the range of retail kinds available. These are unsurprising given that Quito is Ecuador's capital and Guayaquil is the country's largest and most populated metropolis.") 
       st.write("As a result, one might expect Corporacion Favorita to target these major cities with the most diverse store types, as evidenced by the highest counts of store nbrs attributed to those two cities.")       
       st.image("")
       st.image('src/Asset/ML_Comp/img1.png')
   
   elif page_mode=='Results':
       st.header("Results")
       st.write("The problem that was tackled is a regression problem. Three models were used to solve the problem: linear regression, decision tree, and random forest.") 
       st.write("The evaluation metrics used for comparing these models are the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Root Mean Squared Logarithmic Error (RMSLE)")
       st.write("The linear regression model produced an MSE of 0.76, an RMSE of 0.87, and an RMSLE of 0.29. ")
       st.write("The decision tree model produced an MSE of 0.16, an RMSE of 0.40, and an RMSLE of 0.10. Finally, the random forest model produced an MSE of 0.08, an RMSE of 0.28, and an RMSLE of 0.08.")
       st.write("A lower value of MSE, RMSE, and RMSLE indicates better performance of the model.")
       st.write("Based on the evaluation metrics, the random forest model performed the best with an MSE of 0.08, an RMSE of 0.28, and an RMSLE of 0.08.")
       st.write("Therefore, the random forest model is a good choice to report as it provided the best performance among the three models.")
   elif page_mode=='You have not selected': 
       st.write('seleect an option')   
   else:
       st.header("Recommendation")
       st.write("we explored the Favorita Grocery Sales Forecasting dataset. We analyzed the various attributes of the dataset and visualized")
       st.write("the sales patterns over time. We then used various time-series forecasting techniques to build a robust sales forecasting model.") 
       st.write("With this model, we can predict the future sales of different products in different stores and make informed decisions about invenmanagement, pricing, and promotions.")
       st.write("The dataset provides a great opportunity for further analysis and prediction, and we encourage you to explore it further.") 
       st.image('src/Asset/ML_Comp/img1.png')

elif app_mode == 'Prediction':
    st.header("CORPORATION FAVORITA GROCERY STORE SALES PREDICTION APP ")
    st.image('src/Asset/ML_Comp/img1.png')

    # Defining the base containers/ main sections of the app
    header = st.container()
    dataset = st.container()
    features_and_output = st.container()

    form = st.form(key="information", clear_on_submit=True)

    check =st.sidebar.checkbox("Know your columns") 

    if check:
        st.sidebar.markdown(""" 
                    - **store_nbr** identifies the store at which the products are sold.
                    - **family** identifies the type of product sold.
                    - **sales** is the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units(1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
                    - **onpromotion** gives the total number of items in a product family that were being promoted at a store at a given date.
                    - **sales_date** is the date on which a transaction / sale was made
                    - **city** is the city in which the store is located
                    - **state** is the state in which the store is located
                    - **store_type** is the type of store, based on Corporation Favorita's own type system
                    - **cluster** is a grouping of similar stores.
                    - **oil_price** is the daily oil price
                    - **holiday_type** indicates whether the day was a holiday, event day, or a workday
                    - **locale** indicates whether the holiday was local, national or regional.
                    - **transferred** indicates whether the day was a transferred holiday or not.
                    """)
#Defining the base containers/ main sections of the app
header = st.container()
dataset = st.container()
features_and_output = st.container()
# Structuring the dataset section
with dataset:
       check = dataset.checkbox("Preview the dataset")
       dataset.markdown("**This is the dataset of Corporation Favorita**")
       if check:
           dataset.write(train_data.head())
       #dataset.write("View sidebar for information on the columns")
# Defining the list of expected variables
expected_inputs = ["Sales_date",  "family",  "store_nbr", "cluster",  "city",  "state",  "onpromotion",  "dcoilwtico ",  "holiday_type", "locale"]

# List of features to encode
categoricals = ["family", "city", "holiday_type", "locale"]

# List of features to scale
cols_to_scale = ['dcoilwtico']
with features_and_output:
       features_and_output.write(" section below captures your input to be used in predictions")
       features_and_output.subheader("Please Enter the detail")
       col1, col2, col3 = st.columns([1, 3, 3])

# Create the input fields
# st.write("This section captures your input to be used in predictions")
# st.subheader("Please Enter the details")
input_data = {}
col1,col2 = st.columns(2)
with col1:
           input_data['store_nbr'] = st.number_input("store_nbr",step=1)
           input_data['family'] = st.selectbox("products", ['AUTOMOTIVE', 'CLEANING', 'BEAUTY', 'FOODS', 'STATIONERY',
          'CELEBRATION', 'GROCERY', 'HARDWARE', 'HOME', 'LADIESWEAR',
          'LAWN AND GARDEN', 'CLOTHING', 'LIQUOR,WINE,BEER', 'PET SUPPLIES'])
           input_data['onpromotion'] =st.number_input("onpromotion",step=1)
           input_data['state'] = st.selectbox("state", ['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
          'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza',
          'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
          'El Oro', 'Esmeraldas', 'Manabi'])
           input_data['store_type'] = st.selectbox("store_type",['D', 'C', 'B', 'E', 'A'])
           input_data['cluster'] = st.number_input("cluster",step=1)
with col2:
       
           input_data['dcoilwtico'] = st.number_input("dcoilwtico",step=1)
           input_data['year_y'] = st.number_input("year",step=1)
           input_data['month_x'] = st.slider("month",1,12)
           input_data['day_x'] = st.slider("day",1,31)
        #    input_data['end_month_x'] = st.selectbox("end_month",['True','False'])
# Create a DataFrame from user inputs
input_df = pd.DataFrame([input_data])
# Selecting categorical and numerical columns separately
cat_columns = [col for col in input_df.columns if input_df[col].dtype == 'object']
num_columns = [col for col in input_df.columns if input_df[col].dtype != 'object']

# Load preprocessing components
encoder, scaler, num_imputer, cat_imputer = load_preprocessing_components()
# Fitting the imputers with training data
# Funtion to ignore AttributeError: 'SimpleImputer' object has no attribute 'keep_empty_features'
try:
    num_imputer.fit(train_data[num_columns])
except AttributeError as e:
    # Handle the AttributeError and continue
    print("An AttributeError occurred:", e)
    traceback.print_exc()  # Print the traceback for debugging purposes
    print("Continuing with prediction...")

cat_imputer.fit(train_data[cat_columns])

# Apply the imputers
input_df_imputed_cat = cat_imputer.transform(input_df[cat_columns])
input_df_imputed_num = num_imputer.transform(input_df[num_columns])
# Encode the categorical columns
input_encoded_df = pd.DataFrame(encoder.transform(input_df_imputed_cat).toarray(),
                                  columns=encoder.get_feature_names(cat_columns))

# Scale the numerical columns
input_df_scaled = scaler.transform(input_df_imputed_num)
input_scaled_df = pd.DataFrame(input_df_scaled , columns = num_columns)

#joining the cat encoded and num scaled
final_df = pd.concat([input_encoded_df, input_scaled_df], axis=1)
prediction_button = st.button("Predict Sales")
if prediction_button:
    # Load XGBoost model
    dt_model = load_xgb_model()

    # Create a DMatrix from the input DataFrame
    dtest = DMatrix(final_df)  # Convert final_df to DMatrix

    # Make predictions
    predictions = dt_model.predict(dtest)

    # Display the prediction
    st.write(f"The predicted sales are: {predictions[0]}.")
    input_df.to_csv("data.csv", index=False)
    st.table(input_df)