# 🧠 Fraud Detection System – AI Powered 🔍

A machine learning-based web application to detect potential fraudulent transactions in financial datasets. The system combines a **Flask backend** and a **React frontend**, providing an intuitive UI for file uploads, fraud analysis, visual metrics, and PDF report downloads.

🎥 **[Project Demo (Video)](https://drive.google.com/file/d/1d8dgesjiLbdEAr99mHwAw9AOIRSyV-eS/view?usp=sharing)**

---

## 🚀 Features

- 📄 CSV File Upload with drag & drop support  
- 🔎 Fraud Detection using a Random Forest model  
- 📊 Real-time Visual Metrics:
  - Fraud Distribution (Pie Chart)
  - Confusion Matrix (Bar Chart)
  - Accuracy and F1 Score
- 📅 Downloadable PDF Report  
- 🌘 Light/Dark Mode Toggle  
- 📈 Upload Progress Bar & File Preview  

---

## 💠 Tech Stack

**Frontend:**
- React.js
- Tailwind CSS
- Axios
- React Router
- Recharts (for data visualization)
- React Dropzone

**Backend:**
- Flask
- Pandas, NumPy
- Scikit-learn, SMOTE
- Flask-CORS
- ReportLab (PDF generation)

---

## 📁 Project Structure

```
backend/
├── app.py              # Flask API for upload, processing, and report generation
├── uploads/            # Stores uploaded CSV files

frontend/
├── src/
│   ├── App.jsx         # Routing logic
│   ├── pages/          # Upload & Results pages
│   ├── components/     # Charts, Dropzone, SummaryCard
│   ├── index.css       # Tailwind styles
├── package.json        # React project config
```

---

## 📊 How It Works

1. User uploads a CSV file (must contain `isFraud` column).
2. Frontend sends the file to the backend via `POST /upload`.
3. Backend:
   - Reads and preprocesses the file
   - Applies SMOTE for class balancing
   - Trains a Random Forest model
   - Calculates Accuracy, F1 Score, Confusion Matrix
4. Frontend:
   - Displays results via graphs and summary card
   - Allows user to download a detailed report via `GET /download-report`

---

## 🧪 Run the App Locally

### 🔧 Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 💻 Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 📌 Sample CSV Format

Make sure your CSV includes at least the `isFraud` column:

```csv
step,amount,type,isFraud
1,100.00,CASH,0
2,2500.00,TRANSFER,1
```

---

✅ Built with ❤️ for smarter financial security!

