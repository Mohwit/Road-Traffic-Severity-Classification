# Road-Traffic-Severity-Classification

his is a multiclass classification project to classify the severity of road accidents into three categories. The project is based on real-world data, and the dataset is highly imbalanced. The goal of this project is to identify major causes of accidents and classify the severity of accidents.

## Dataset
The data set is collected from Addis Ababa Sub-city, Ethiopia police departments for a master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. The dataset has 32 features and 12,316 instances of the accident. The target feature is "Accident_severity", which is a multi-class variable.

## Web application
The streamlit app is used for deploying the model.

## Tasks and techniques used:
1. Exploratory data analysis
    - Data analysis using dabl
    - Exploratory data analysis using matplotlib and seaborn

2. Data preparation and pre-processing
    - Missing Values Treatment using fillna method
    - One Hot encoding using pandas get_dummies
    - Feature selection using chi2 statistic and SelectKBest method
    - PCA to reduce dimensionality
    - Imbalance data treatment using SMOTENC technique

3. Modelling using sci-kit learn library
    - Baseline model using RandomForest using default technique
    - Tuned hyperparameters using n_estimators and max_depth parameters

4. Evaluation
    - Evaluation metric was weighted f1_score
    - Baseline model evaluation f1_score = 61%
    - Final model evaluation f1_score = 88%
    
## Running the Project
To get started with the project, you can simply clone the repository using the following command:

`
git clone  https://github.com/yourusername/Road-Traffic-Severity-Classification.git
`

After cloning the repository, install the required packages using the following command:

`
pip install -r requirements.txt
`

Run the streamlit application:

`
streamlit run app.py
`

## Conclusion
The project was able to identify the major causes of accidents and classify the severity of accidents. The final model was able to achieve an f1-score of 88%, which is a significant improvement over the baseline model. This project can be useful for policymakers and stakeholders to identify major causes of accidents and take necessary actions to reduce accidents.
