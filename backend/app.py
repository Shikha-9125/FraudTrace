# import os  # accessing directory structure
# import math  # mathematical functions
# import numpy as np  # linear algebra
# import pandas as pd  # data processing, CSV file I/O (e.g., pd.read_csv)
# import matplotlib.pyplot as plt  # plotting
# import seaborn as sns  # for enhanced visualization
# from mpl_toolkits.mplot3d  import Axes3D
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# # Specify the path to your dataset
# file_path = "C:/Users/anita/OneDrive/Desktop/minor/FraudDetection/backend/PS_20174392719_1491204439457_log.csv"

# # Check if the file exists
# if os.path.isfile(file_path):
#     print("File exists!")
# else:
#     print(f"File does not exist: {file_path}")

# # Load your dataset
# nRowsRead = 1000  # specify 'None' if want to read the whole file
# df1 = pd.read_csv(file_path, delimiter=',', nrows=nRowsRead)
# df1.dataframeName = 'PS_20174392719_1491204439457_log.csv'
# nRow, nCol = df1.shape
# print(f'There are {nRow} rows and {nCol} columns')

# # Display the first few rows of the DataFrame
# print(df1.head(5))

# # Check the class distribution
# print(df1['isFraud'].value_counts())

# # Distribution graphs (histogram/bar graph) of column data
# def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
#     nunique = df.nunique()
#     df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]  # For displaying purposes
#     nRow, nCol = df.shape
#     columnNames = list(df)
#     nGraphRow = math.ceil(nCol / nGraphPerRow)
#     plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
        
#     for i in range(min(nCol, nGraphShown)):
#         plt.subplot(nGraphRow, nGraphPerRow, i + 1)
#         columnDf = df.iloc[:, i]
#         if not np.issubdtype(type(columnDf.iloc[0]), np.number):
#             valueCounts = columnDf.value_counts()
#             valueCounts.plot.bar()
#         else:
#             columnDf.hist(bins=30)  # Added bins for better histogram representation
#         plt.ylabel('counts')
#         plt.xticks(rotation=90)
#         plt.title(f'{columnNames[i]} (column {i})')
#     plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
#     plt.show()

# # Correlation matrix
# def plotCorrelationMatrix(df, graphWidth):
#     filename = df.dataframeName
#     df = df.dropna(axis=1)  # drop columns with NaN
#     df = df.select_dtypes(include=[np.number])  # keep only numeric columns
#     df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns with more than 1 unique value
#     if df.shape[1] < 2:
#         print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
#         return
#     corr = df.corr()
#     plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
#     corrMat = plt.matshow(corr, fignum=1)
#     plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
#     plt.yticks(range(len(corr.columns)), corr.columns)
#     plt.gca().xaxis.tick_bottom()
#     plt.colorbar(corrMat)
#     plt.title(f'Correlation Matrix for {filename}', fontsize=15)
#     plt.show()

# # Scatter and density plots
# def plotScatterMatrix(df, plotSize, textSize):
#     df = df.select_dtypes(include=[np.number])  # keep only numerical columns
#     df = df.dropna(axis=1)  # drop columns with NaN
#     df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns with more than 1 unique value
#     columnNames = list(df)
#     if len(columnNames) > 10:  # reduce the number of columns for kernel density plots
#         columnNames = columnNames[:10]
#     df = df[columnNames]
#     ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
#     corrs = df.corr().values
#     for i, j in zip(*np.triu_indices_from(ax, k=1)):
#         ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
#     plt.suptitle('Scatter and Density Plot')
#     plt.show()

# # Prepare the data for resampling
# y = df1['isFraud']  # This is your target variable
# X_numeric = df1.select_dtypes(include=[float, int])

# # Check for NaN values in the numeric DataFrame
# if X_numeric.isnull().sum().any():
#     print("NaN values detected in the numeric features. Handling NaN values...")
#     # Fill NaN with mean of each column
#     X_numeric.fillna(X_numeric.mean(), inplace=True)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42, stratify=y)

# # Create an instance of SMOTE
# smote = SMOTE(sampling_strategy='auto', random_state=42)

# # Fit and resample the training data
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# # Check the shapes of the resampled datasets
# print(f"Original training set shape: {X_train.shape}, {y_train.shape}")
# print(f"Resampled training set shape: {X_resampled.shape}, {y_resampled.shape}")

# # Train a basic Random Forest model
# model = RandomForestClassifier(random_state=42)
# model.fit(X_resampled, y_resampled)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Print the classification report
# print(classification_report(y_test, y_pred))

# # Confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
# plt.figure(figsize=(8, 6))
# plt.title('Confusion Matrix')
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

# # Call the plotPerColumnDistribution function
# plotPerColumnDistribution(df1, 10, 5)

# # Call the plotCorrelationMatrix function
# plotCorrelationMatrix(df1, 8)

# # Call the plotScatterMatrix function
# plotScatterMatrix(df1, 20, 10)
# import os
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# from flask_cors import CORS   # <-- import flask-cors
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix

# # --- Flask App Setup ---
# app = Flask(__name__)
# CORS(app)  # <-- Enable CORS for all routes (allow frontend to call backend)
# UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ALLOWED_EXTENSIONS = {'csv'}

# # Create uploads folder if not exists
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # --- Core ML logic wrapped into a function ---
# def process_file(file_path):
#     try:
#         df = pd.read_csv(file_path, delimiter=',', nrows=1000)
#         df.dataframeName = os.path.basename(file_path)

#         # ML pipeline
#         y = df['isFraud']
#         X_numeric = df.select_dtypes(include=[float, int])
#         X_numeric.fillna(X_numeric.mean(), inplace=True)

#         X_train, X_test, y_train, y_test = train_test_split(
#             X_numeric, y, test_size=0.2, random_state=42, stratify=y
#         )

#         smote = SMOTE(sampling_strategy='auto', random_state=42)
#         X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

#         model = RandomForestClassifier(random_state=42)
#         model.fit(X_resampled, y_resampled)
#         y_pred = model.predict(X_test)

#         report = classification_report(y_test, y_pred, output_dict=True)
#         conf_matrix = confusion_matrix(y_test, y_pred).tolist()

#         return {
#             'rows': df.shape[0],
#             'cols': df.shape[1],
#             'isFraud_counts': df['isFraud'].value_counts().to_dict(),
#             'classification_report': report,
#             'confusion_matrix': conf_matrix
#         }

#     except Exception as e:
#         return {'error': str(e)}

# # --- Upload Route ---
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         result = process_file(file_path)
#         return jsonify(result), 200

#     return jsonify({'error': 'Invalid file type'}), 400

# # --- Run App ---
# if __name__ == '__main__':
#     app.run(debug=True)
import os
import pandas as pd
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

# Global variable to store report
after_upload_report = {}

# Create uploads folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Core ML logic wrapped into a function ---
def process_file(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',', nrows=1000)
        df.dataframeName = os.path.basename(file_path)

        # ML pipeline
        y = df['isFraud']
        X_numeric = df.select_dtypes(include=[float, int])
        X_numeric.fillna(X_numeric.mean(), inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=0.2, random_state=42, stratify=y
        )

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Save the report in the global variable after_upload_report
        after_upload_report['classification_report'] = report
        after_upload_report['confusion_matrix'] = conf_matrix

        return {
            'rows': df.shape[0],
            'cols': df.shape[1],
            'isFraud_counts': df['isFraud'].value_counts().to_dict(),
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }

    except Exception as e:
        return {'error': str(e)}

# --- Upload Route ---
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result = process_file(file_path)
        return jsonify(result), 200

    return jsonify({'error': 'Invalid file type'}), 400

# --- Report Download Route ---
@app.route('/download-report', methods=['GET'])
def download_report():
    if not after_upload_report:
        return jsonify({'error': 'No recent report available'}), 400

    # Generate PDF from the report in after_upload_report
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    # Title
    c.setFont("Helvetica", 16)
    c.drawString(100, 750, "Fraud Detection Report")

    # Classification Report
    c.setFont("Helvetica", 12)
    y_position = 730
    classification_report = after_upload_report['classification_report']
    for label, metrics in classification_report.items():
        if isinstance(metrics, dict):
            c.drawString(100, y_position, f"Class: {label}")
            y_position -= 15
            for metric, value in metrics.items():
                c.drawString(120, y_position, f"{metric}: {value}")
                y_position -= 15
        elif label != 'accuracy':
            c.drawString(100, y_position, f"{label}: {metrics}")
            y_position -= 15

    # Confusion Matrix
    c.drawString(100, y_position, "Confusion Matrix:")
    y_position -= 20
    confusion_matrix = after_upload_report['confusion_matrix']
    for i, row in enumerate(confusion_matrix):
        c.drawString(100, y_position, f"Row {i + 1}: {row}")
        y_position -= 15

    c.showPage()
    c.save()

    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="fraud_report.pdf", mimetype="application/pdf")

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
