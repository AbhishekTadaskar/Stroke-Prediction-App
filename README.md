# 🧠 Stroke Prediction App

## 📋 Overview
This project is a machine learning application that predicts the likelihood of a patient having a stroke based on various health metrics. The process involves training and evaluating several machine learning models on a provided healthcare dataset, identifying the best-performing model, and then deploying it within an interactive web application built with Streamlit.

---

## 🗂️ File Structure
- **`healthcare-dataset-stroke-data.csv`** 📄: The primary dataset used for model training and evaluation.  
- **`model_evaluation_and_tuning.py`** 🧪: Performs data preprocessing, trains and evaluates multiple machine learning models (Decision Tree, Random Forest, AdaBoost, XGBoost, CatBoost), and saves the best-performing model as a pickle file if it meets a performance threshold.  
- **`best_model.pkl`** 💾: The serialized machine learning model, created by `model_evaluation_and_tuning.py`, which is loaded by the Streamlit application for making predictions.  
- **`streamlit_app.py`** 🌐: The Python script for the interactive web application. It takes user input and uses the `best_model.pkl` file to generate a stroke likelihood prediction.  
- **`requirements.txt`** 📦: A list of all Python libraries required to run the project.  
- **`README.md`** 📝: This file, providing an overview of the project.

---

## 🏗️ Project Architecture
The project's architecture is a streamlined, three-step pipeline designed for efficient model development and deployment.

```mermaid
flowchart TD
    subgraph Data & Training Pipeline
        A[📄 healthcare-dataset-stroke-data.csv] -->|Reads & Preprocesses Data| B[🧪 model_evaluation_and_tuning.py]
        B -->|Trains & Evaluates Models| C[💾 best_model.pkl]
    end
    
    subgraph Prediction Service
        C -->|Loads Model| D[🌐 streamlit_app.py]
        D -- "Accepts user input" --> E((👤 User Input))
        E -- "Feeds data to model" --> D
        D -- "Returns prediction" --> F((✅ Prediction Result))
    end
    
    A & B & C & D & E & F
