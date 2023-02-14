## importing libraries
import numpy as np
import streamlit as st
import joblib
import shap
from PIL import Image

# 1: serious injury, 2: Slight injury, 0: Fatal Injury

## getting our trained model
model = joblib.load("rta_model_deploy3.joblib")
encoder = joblib.load("ordinal_encoder2.joblib")

## setting page configuration
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Accident Severity Prediction App", page_icon="🚧", layout="wide")

## creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

# # hour of the day: range of 0 to 23
options_types_collision = ['Vehicle with vehicle collision' ,'Collision with roadside objects',
                           'Collision with pedestrians' ,'Rollover' ,'Collision with animals',
                           'Unknown' ,'Collision with roadside-parked vehicles' ,'Fall from vehicles',
                           'Other' ,'With Train']
## option's for sex
options_sex = ['Male' ,'Female' ,'Unknown']

## for education levels
options_education_level = ['Junior high school' ,'Elementary school' ,'High school',
                           'Unknown' ,'Above high school' ,'Writing & reading' ,'Illiterate']

## options for service years
options_services_year = ['Unknown' ,'2-5yrs' ,'Above 10yr' ,'5-10yrs' ,'1-2yr' ,'Below 1yr']

## options for accident area
options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',' Industrial areas',
                    'School areas', 'Recreational areas', ' Outside rural areas', ' Hospital areas',
                    'Market areas', 'Rural village areas', 'Unknown', 'Rural village areas Office areas',
                    'Recreational areas']

# features list
features = ['Number_of_vehicles_involved' ,'Number_of_casualties' ,'Hour_of_Day' ,'Type_of_collision'
            ,'Age_band_of_driver' ,'Sex_of_driver', 'Educational_level' ,'Service_year_of_vehicle' ,
            'Day_of_week' ,'Area_accident_occurred']
#
# # take input
st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    a, b, c = st.columns([0.1, 0.8, 0.1])
    with b:
        st.subheader("Pleas enter the following inputs:")
        with st.form("road_traffic_severity_form"):
            ## number of vehicle involved: range of 1 to 7
            No_vehicles = st.slider("Number of vehicles involved:" ,1 ,7, value=0, format="%d")

            ## number of casualties: range of 1 to 8
            No_casualties = st.slider("Number of casualties:" ,1 ,8, value=0, format="%d")

            ## hour of the day: range of 0 to 23
            Hour = st.slider("Hour of the day:", 0, 23, value=0, format="%d")

            ## collision
            collision = st.selectbox("Type of collision:" ,options=options_types_collision)

            ## age
            Age_band = st.selectbox("Driver age group?:", options=options_age)

            ## sex
            Sex = st.selectbox("Sex of the driver:", options=options_sex)

            ## education level
            Education = st.selectbox("Education of driver:" ,options=options_education_level)

            ## vehicle servie years
            service_vehicle = st.selectbox("Service year of vehicle:", options=options_services_year)

            ## day
            Day_week = st.selectbox("Day of the week:", options=options_day)

            ## accident area
            Accident_area = st.selectbox("Area of accident:", options=options_acc_area)

            ## submit button
            submit = st.form_submit_button("Predict")
    #
        # encode using ordinal encoder and predict
        if submit:

            ## converting the option taken to array
            input_array = np.array([collision,
                                    Age_band ,Sex ,Education ,service_vehicle,
                                    Day_week ,Accident_area], ndmin=2)

            ## encoding the inputs
            encoded_arr = list(encoder.transform(input_array).ravel())

            ## numerical inputs
            num_arr = [No_vehicles ,No_casualties ,Hour]

            ## creating the final array to pass to model
            ## adding the numerical array
            pred_arr = np.array(num_arr + encoded_arr).reshape(1 ,-1)

            ## making prediction
            prediction = model.predict(pred_arr)

            ## if prediction is 0
            if prediction == 0:
                st.write(f"The severity prediction is Fatal Injury⚠")

            ## if prediction is 1
            elif prediction == 1:
                st.write(f"The severity prediction is serious injury")

            ## if prediction is 2
            else:
                st.write(f"The severity prediction is slight injury")

            ## Using shap
            st.subheader("Explainable AI (XAI) to understand predictions")
            shap.initjs()
            shap_values = shap.TreeExplainer(model).shap_values(pred_arr)

            ## for predicted value
            st.write(f"For prediction {prediction}")

            ## plotting / displaying the image
            shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0],
                            pred_arr, feature_names=features, matplotlib=True ,show=False).savefig("pred_force_plot.jpg", bbox_inches='tight')
            img = Image.open("pred_force_plot.jpg")
            st.image(img, caption='Model explanation using shap')


# post the image of the accident
a ,b ,c = st.columns([0.2 ,0.6 ,0.2])
st.markdown("""---""")
## displaying a picture in starting
with b:
    st.markdown("""---""")
    ## displaying image
    st.image("road_traffic_accident.jpg", use_column_width=True)

## calling main function
if __name__ == '__main__':
   main()
