#Importing the dependencies
import os
import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
import streamlit as st
from streamlit_option_menu import option_menu
import base64
import pickle as pk
import joblib
# from catboost import CatBoostClassifier
# import option_menu




#configuring the page setup
st.set_page_config(page_title='Stroke-prediction system',layout='centered')

with st.sidebar:
    st.title("Home Page")
    selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction"],icons=["cast","book","cast"],menu_icon="house",default_index=0)


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your prediction</a>'
    return href




def obesity_recommendation(prediction):
    if prediction == 0 or prediction == "0":
        return (
            "Insufficient Weight:\n"
            "Recommendation:\n"
            "1. Eat a balanced diet rich in protein, healthy fats, and complex carbs.\n"
            "2. Engage in strength training to build healthy muscle mass.\n"
            "3. Consult a nutritionist if needed to gain weight safely."
        )

    elif prediction == 1 or prediction == "1":
        return (
            "Normal Weight:\n"
            "Recommendation:\n"
            "1. Maintain a balanced diet.\n"
            "2. Engage in regular physical activity.\n"
            "3. Schedule routine health check-ups to stay healthy."
        )

    elif prediction == 2 or prediction == "2":
        return (
            "Overweight Level I:\n"
            "Recommendation:\n"
            "1. Focus on a healthy, calorie-controlled diet.\n"
            "2. Increase physical activity with regular exercise like walking or light cardio.\n"
            "3. Monitor your weight regularly and set achievable goals."
        )

    elif prediction == 3 or prediction == "3":
        return (
            "Overweight Level II:\n"
            "Recommendation:\n"
            "1. Adopt a structured diet plan.\n"
            "2. Increase exercise intensity and reduce processed foods.\n"
            "3. Consider consulting a dietitian for personalized guidance."
        )

    elif prediction == 4 or prediction == "4":
        return (
            "Obesity Type I:\n"
            "Recommendation:\n"
            "1. Start a comprehensive weight management plan including diet and regular exercise.\n"
            "2. Implement lifestyle changes like better sleep and stress management.\n"
            "3. Have regular health check-ups for blood pressure, blood sugar, and cholesterol."
        )

    elif prediction == 5 or prediction == "5":
        return (
            "Obesity Type II:\n"
            "Recommendation:\n"
            "1. Seek guidance from a healthcare provider for a tailored weight loss plan.\n"
            "2. Follow a structured diet and exercise program under supervision.\n"
            "3. Monitor health parameters and check for obesity-related complications."
        )

    elif prediction == 6 or prediction == "6":
        return (
            "Obesity Type III:\n"
            "Recommendation:\n"
            "1. Consult a healthcare provider for a personalized treatment plan.\n"
            "2. Focus on medical guidance, diet therapy, and supervised physical activity.\n"
            "3. Monitor for obesity-related complications and manage them promptly."
        )

    else:
        return "Invalid prediction value. Please check the input."




def obesity_predict_only(prediction):
    if prediction == 0 or prediction == "0":
        return (
            "Insufficient Weight"
        )

    elif prediction == 1 or prediction == "1":
        return (
            "Normal Weight"
        )

    elif prediction == 2 or prediction == "2":
        return (
            "Overweight Level I"
           
        )

    elif prediction == 3 or prediction == "3":
        return (
            "Overweight Level II"
        )

    elif prediction == 4 or prediction == "4":
        return (
            "Obesity Type I"
        )

    elif prediction == 5 or prediction == "5":
        return (
            "Obesity Type II"
        )

    elif prediction == 6 or prediction == "6":
        return (
            "Obesity Type III"
        )

    else:
        return "Invalid prediction value. Please check the input."


def encode_ordinal(df, column, categories):
    # make sure it's string, remove spaces, normalize case
    df[column] = df[column].astype(str).str.strip()

    enc = OrdinalEncoder(categories=categories)
    df[column] = enc.fit_transform(df[[column]])
    return df

#single prediction function
def obesity_detect(givendata):

    loaded_model=pk.load(open("ObesityModel.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    std_scaler_loaded=pk.load(open("obesityscaler.pkl", "rb"))
    std_X_resample=std_scaler_loaded.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_X_resample)
    solution=obesity_predict_only(prediction)
    return solution
    

#main function handling the input
def main():
    st.header("ObesityOracle (Obesity predictive system)")
    
    #getting user input
    
    age = st.number_input('What is your age',format=None, key="age")
    st.write("Patient's age is :", age, 'years old')

    gender = st.selectbox('Gender',("",'Male' ,'Female'),key="sex")
    if (gender=='Male'):
        gender=1
    else:
        gender=0

    option4 = st.number_input('Height',format=None, key="height")
    st.write("Your height is :", option4,)

    option5 = st.number_input('Weight',format=None, key="weight")
    st.write("Your weight is :", option5,)


    alcohol = st.selectbox("Do you consume alcohol ?",("","no", "Sometimes", "Frequently", "Always"),key="alcohol")
    if alcohol== "no":
        alcohol= 0
    if alcohol== "Sometimes":
        alcohol= 1
    if alcohol== "Frequently":
        alcohol= 2
    else:
        alcohol=3

    caloric = st.selectbox("Frequent consumption of high caloric food ? ",("","yes", "no"),key="caloricfood")
    if caloric== "yes":
        caloric=1
    else:
        caloric=0


    option7 = st.number_input('Frequency of consumption of vegetables',format=None, key="veg")
    st.write("Your frequesnt consumption is :", option7,)


    calorieintake = st.selectbox("Do you monitor your calorie intake ? ",("","yes", "no"),key="calorieintake")
    if calorieintake== "yes":
        calorieintake=1
    else:
        calorieintake=0


    waterconsumption = st.number_input('Daily water consumption measures',format=None, key="water")
    st.write('Your average waterconsumption level is ', waterconsumption)


    foodbetweenmeals = st.selectbox("How often do you eat food between meals (snacking)?",("","no", "Sometimes", "Frequently", "Always"),key="foodbtween")
    if foodbetweenmeals== "no":
        foodbetweenmeals= 0
    if foodbetweenmeals== "Sometimes":
        foodbetweenmeals= 1
    if foodbetweenmeals== "Frequently":
        foodbetweenmeals= 2
    else:
        foodbetweenmeals=3


    family_history_with_overweight = st.selectbox('Family history Overweight/Obesity',("","yes", "no",),key="familyhistory")
    if (family_history_with_overweight=='yes'):
        family_history_with_overweight=1

    else:
        family_history_with_overweight=0

    
    physicalactivity = st.number_input('what is the frequency of your physical activities',format=None, key="physicalactivity")
    st.write('Your physical activity frequency is ', physicalactivity)

    electronicdevice = st.number_input('what is the frequency of Time using technology devices',format=None, key="electronicactivity")
    st.write('The frequency of your time using technology devices', electronicdevice)


    transportation = st.selectbox("What is your means of transportation ?",("","Public_Transportation", 'Walking', 'Automobile', 'Motorbike','Bike'),key="transportation")
    if transportation== "Public_Transportation":
        transportation= 0
    if transportation== "Sometimes":
        transportation= 1
    if transportation== "Frequently":
        transportation= 2
    else:
        transportation=3

    st.write("\n")
    st.write("\n")

    detectionResult = '' #for displaying result
    
    # creating a button for Prediction
    if age!="" and gender!="" and option4!="" and option5!="" and caloric!="" and option7 !="" and alcohol !="" and waterconsumption!="" and calorieintake!="" and family_history_with_overweight!="" and st.button('Predict'):
        detectionResult = obesity_detect([age,gender,option4,option5, alcohol,caloric,option7, calorieintake, waterconsumption,family_history_with_overweight, physicalactivity,electronicdevice,foodbetweenmeals, transportation])
        st.success(detectionResult)



# def multi(input_data):
#     loaded_model=pk.load(open("ObesityModel.sav", "rb"))
#     dfinput = pd.read_csv(input_data)
#     if "SMOKE" or "smoke" in dfinput.iloc[1:]:
#         dfinput.drop("SMOKE",axis=1,inplace=True)
#     if "NCP" or "ncp" in dfinput.iloc[1:]:
#         dfinput.drop("NCP",axis=1,inplace=True)
#     if "NObeyesdad" in dfinput.iloc[1:]:
#         dfinput.drop("NObeyesdad",axis=1,inplace=True)
    
#     dfinput=dfinput.drop(dfinput.columns[0],axis=1)
#     dfinput=dfinput.reset_index(drop=True)

#     st.header('A view of your uploaded dataset')
#     st.markdown('')
#     st.dataframe(dfinput)

#     dfinput=dfinput.values
#     std_scaler_loaded=pk.load(open("obesityscaler.pkl", "rb"))
#     std_dfinput=std_scaler_loaded.transform(dfinput)
    
    
#     predict=st.button("predict")


#     if predict:
#         prediction = loaded_model.predict(std_dfinput)
#         interchange=[]
#         condition=obesity_predict_only(prediction)
#         interchange.append(condition)
#         # for i in prediction:
#         #     if i==1:
#         #         newi="Stroke issues present"
#         #         interchange.append(newi)
#         #     elif i==0:
#         #         newi="No Stroke issues"
#         #         interchange.append(newi)
            
#         st.subheader('All the predictions')
#         prediction_output = pd.Series(interchange, name='Obesity prediction results')
#         prediction_id = pd.Series(np.arange(len(interchange)),name="User_ID")
#         dfresult = pd.concat([prediction_id, prediction_output], axis=1)
#         st.dataframe(dfresult)
#         st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        
def encode_ordinal(df, column, categories):
    enc = OrdinalEncoder(categories=categories)
    df[column] = enc.fit_transform(df[[column]])
    return df

def multi(input_data):
    loaded_model = pk.load(open("ObesityModel.sav", "rb"))
    dfinput = pd.read_csv(input_data)

    # Drop unused cols if they exist
    for col in ["SMOKE", "NCP", "NObeyesdad"]:
        if col in dfinput.columns:
            dfinput.drop(col, axis=1, inplace=True)

    # Drop first index column if exists
    if dfinput.columns[0].lower() in ["unnamed: 0", "id"]:  # common auto index names
        dfinput = dfinput.drop(dfinput.columns[0], axis=1)
    dfinput = dfinput.reset_index(drop=True)

    # Detect if dataset has categorical/string columns to preprocess
    string_cols = dfinput.select_dtypes(include=["object"]).columns.tolist()
    if string_cols:
        # Yes/no columns
        yes_no_cols = ["FAVC", "SCC", "family_history_with_overweight", "SMOKE", "Gender"]
        for col in yes_no_cols:
            if col in dfinput.columns:
                dfinput[col] = dfinput[col].astype(str).str.strip().str.lower()
                if col == "Gender":
                    dfinput[col] = dfinput[col].replace({"male": 1, "female": 0}).astype("int64")
                else:
                    dfinput[col] = dfinput[col].replace({"no": 0, "yes": 1}).astype("int64")

        # CAEC / CALC ordinal
        if "CAEC" in dfinput.columns:
            dfinput["CAEC"] = dfinput["CAEC"].astype(str).str.strip().str.lower()
            caec_order = [["no", "sometimes", "frequently", "always"]]
            dfinput = encode_ordinal(dfinput, "CAEC", caec_order)

        if "CALC" in dfinput.columns:
            dfinput["CALC"] = dfinput["CALC"].astype(str).str.strip().str.lower()
            calc_order = [["no", "sometimes", "frequently", "always"]]
            dfinput = encode_ordinal(dfinput, "CALC", calc_order)

        # MTRANS
        if "MTRANS" in dfinput.columns:
            dfinput["MTRANS"] = dfinput["MTRANS"].str.strip().str.lower()
            dfinput["MTRANS"] = dfinput["MTRANS"].replace({
                "public_transportation": 0,
                "walking": 1,
                "automobile": 2,
                "motorbike": 3,
                "bike": 4
            }).astype("int64")

    # Convert to numpy
    dfinput_values = dfinput.values

    # Standard scaler
    std_scaler_loaded = pk.load(open("obesityscaler.pkl", "rb"))
    std_dfinput = std_scaler_loaded.transform(dfinput_values)

    # Show table
    st.header('A view of your dataset')
    st.dataframe(pd.DataFrame(dfinput_values, columns=dfinput.columns))

    # Predict button
    predict = st.button("Predict")
    if predict:
        prediction = loaded_model.predict(std_dfinput)
        interchange = [obesity_predict_only(prediction)]
        st.subheader('All the predictions')
        prediction_output = pd.Series(interchange, name='Obesity prediction results')
        prediction_id = pd.Series(np.arange(len(interchange)), name="User_ID")
        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(dfresult)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)



if selection == "Single Prediction":
    main()

# if selection == "Multi Prediction":
#     #st.set_option('deprecation.showPyplotGlobalUse', False)
#     #---------------------------------#
#     # Prediction
#     #--------------------------------
#     #---------------------------------#
#     # Sidebar - Collects user input features into dataframe
#     st.header('Upload your csv file here')
#     uploaded_file = st.file_uploader("", type=["csv","xls"])
#     #--------------Visualization-------------------#
#     # Main panel
    
#     # Displays the dataset
#     if uploaded_file is not None:
#         #load_data = pd.read_table(uploaded_file).
#         multi(uploaded_file)
#     else:
#         st.info('Upload your dataset !!')
    


if selection == "Multi Prediction":
    st.header("Upload your csv or excel file here")

    uploaded_file = st.file_uploader("", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        ext = os.path.splitext(file_name)[1]

        if ext not in [".csv", ".xls", ".xlsx"]:
            st.error("❌ Invalid file type. Please upload a CSV or Excel file (.csv, .xls, .xlsx).")
        else:
            multi(uploaded_file)

    else:
        st.info("Upload your dataset !!")


