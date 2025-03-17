#Doctor-Survey-Prediction-System
Project Overview

This web application predicts the best doctors (NPIs)
 to target for a survey based on their login activity, time spent, and attempts.
 Instead of sending survey invitations to all doctors, 
 this app uses Machine Learning (Random Forest & Logistic Regression with ensembling)
 to predict the doctors most likely to respond at a given time.

The user inputs a specific time, 
and the app generates a downloadable CSV file containing 
the NPIs of the doctors most likely to participate in the survey.

Dataset
The dataset contains dummy NPIs with login/logout times, survey participation, and user behavior.


Technologies Used
>Python (Flask, Pandas, NumPy, Scikit-learn)
>Machine Learning (Random Forest, Logistic Regression, Ensembling)
>Frontend (HTML, CSS, Bootstrap)
>Data Processing (Pandas, OpenPyXL for Excel files)
>Deployment(GitHub, Render)

WEBSITE LINK -https://doc-predict-2.onrender.com/
