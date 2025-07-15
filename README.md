
# Income Prediction Using Machine Learning Models

This project implements a machine learning model to predict whether an individual's income exceeds $50K per year based on demographic features. The project compares multiple machine learning models, including **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**. Hyperparameter tuning is performed on each model to optimize its performance. The final model is deployed using **Streamlit** for real-time predictions.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Model Performance](#model-performance)
7. [Challenges Faced](#challenges-faced)
8. [Future Enhancements](#future-enhancements)

---

## Project Overview

This project focuses on predicting whether a person earns more than $50K per year based on a dataset of census attributes such as age, education, and occupation. The solution involves the following steps:

1. **Data Preprocessing**: Cleaning, scaling, and transforming the raw data.
2. **Model Building**: Building multiple machine learning models and comparing their performances.
3. **Hyperparameter Tuning**: Optimizing the models using **GridSearchCV** and **RandomizedSearchCV**.
4. **Model Evaluation**: Evaluating the models based on **accuracy**, **precision**, **recall**, and **F1 score**.
5. **Deployment**: Deploying the best model using **Streamlit** for real-time predictions.

---

## Technologies Used

- **Programming Language**: Python
- **Machine Learning Libraries**:
  - **Scikit-learn**: For building and evaluating models.
  - **XGBoost**: For implementing the XGBoost model.
  - **SHAP**: For model interpretability (SHAP values).
- **Data Manipulation and Visualization**:
  - **Pandas**: For data preprocessing.
  - **NumPy**: For numerical operations.
  - **Seaborn** and **Matplotlib**: For data visualization and plotting charts.
- **Deployment**:
  - **Streamlit**: For deploying the model as a web application.
- **Model Optimization**:
  - **GridSearchCV** and **RandomizedSearchCV**: For hyperparameter tuning.

---

## Setup Instructions

To set up the project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/income-prediction.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd income-prediction
   ```

3. **Create and activate a virtual environment**:
   - For **Windows**:
     ```bash
     python -m venv venv
     .env\Scriptsctivate
     ```
   - For **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Once you have the environment set up, you can run the Streamlit app to make predictions:

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. The app will open in your default web browser at `http://localhost:8501/`.

3. **Input Features**:
   - Enter values for `age`, `education.num`, `capital.gain`, `capital.loss`, and `hours.per.week`.
   - Click the "Predict Income" button to see the predicted income category (`<=50K` or `>50K`).

---

## File Structure

Here is a brief overview of the project structure:

```
income-prediction/
├── app/
│   └── app.py 
├── data/                          # Folder containing the raw dataset
│   └── adult.csv                  # The dataset used for training
├── app.py                         # Streamlit app for deployment
├── Models.ipnyb                   # Code for training models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
├── requirements.txt               # List of required Python packages
├── Income_Prediction_Report.docx  # Project report document
├── SUMMER-TRAINING-Project.pptx   # Presentation on the project
└── README.md                      # Project overview and setup instructions
```

---

## Model Performance

### Evaluation Metrics:
- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: Indicates the percentage of positive predictions that were correct.
- **Recall**: Measures the percentage of actual positives correctly identified.
- **F1 Score**: The harmonic mean of precision and recall.

### Performance Summary:

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 85.3%    | 75%       | 62%    | 0.68     |
| Decision Tree         | 85.6%    | 80%       | 56%    | 0.66     |
| Random Forest         | 85.5%    | 70%       | 73%    | 0.72     |
| XGBoost              | 87.4%    | 79%       | 67%    | 0.72     |

---

## Challenges Faced

1. **Handling Missing Data**: The dataset contained missing values, which required careful handling and imputation.
2. **Hyperparameter Tuning**: The tuning process was time-consuming, especially for **Random Forest** and **XGBoost**, as these models have many hyperparameters.
3. **Feature Selection**: Deciding which features to include for training required trial and error, as the dataset had many features.

---

## Future Enhancements

1. **Model Improvement**:
   - Experiment with other models like **Gradient Boosting** or **SVM** to further improve performance.
   - Tune the models further by exploring different hyperparameters or using ensemble methods.

2. **Real-time Data**:
   - Integrate real-time data collection using an API or user feedback to continuously update the model with new data.

3. **Deploy on Cloud**:
   - Host the **Streamlit app** on platforms like **Heroku** or **AWS** for production use.

