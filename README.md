# ML Laboratory: Which Algorithm Thinks Like What?

An interactive Streamlit app for comparing machine-learning algorithms.
Upload a CSV dataset and the app automatically:

- Detects and handles missing values
- Preprocesses the data (imputation, encoding, scaling)
- Trains KNN, Decision Tree, SVM and Logistic Regression
- Tunes hyperparameters with GridSearchCV
- Compares accuracy, precision, recall, F1 and cross-validation scores
- Visualises decision boundaries using PCA
- Explains where each algorithm succeeds or fails

## Tech stack
Python, Streamlit, scikit-learn, pandas, NumPy, Matplotlib

## How to run
```
pip install -r requirements.txt
streamlit run appp.py
```
Then open http://localhost:8501 in your browser.

## How to use
1. Upload a CSV file or choose a sample dataset in the sidebar.
2. Select the target column.
3. Click **Run experiment**.
4. Explore the tabs: data and missing values, preprocessing, metrics,
   decision boundaries and explanation.

## Algorithms compared
| Algorithm | Idea |
|---|---|
| KNN | Classifies by voting among the nearest points |
| Decision Tree | Asks a sequence of yes/no questions on features |
| SVM | Finds the widest-margin boundary between classes |
| Logistic Regression | Draws a straight line and outputs probabilities |


