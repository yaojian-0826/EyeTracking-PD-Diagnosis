# Eye Movement SHAP Calculator

An interactive **Streamlit** web application for **Parkinson's disease classification** using **Random Forest + SHAP** explanation on eye-movement features.

## 🌐 Live Demo

Deployed on Streamlit Community Cloud:
```
https://YOUR_USERNAME-eyemove-shap-app-xxxxx.streamlit.app
```

## 📁 Project Structure

```
shap-eyemove-app/
├── app.py                          ← Main Streamlit application
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
├── shap解释测试集.xlsx              ← Training dataset (141 samples × 5 features)
└── shap解释验证集.xlsx              ← Validation dataset (59 samples × 5 features)
```

## 🧠 Features

### 5 Eye Movement Features
| Feature | Description |
|---------|-------------|
| Visuospatial Executive Function | Cognitive function score |
| Attention | Attention function score |
| Mean Overlap Saccade Velocity | Saccadic control (deg/s) |
| Mean Anti-Saccade Velocity | Inhibitory control (deg/s) |
| Orientation | Orientation function score |

### 4 Pages
| Page | Content |
|------|---------|
| 🔮 **Predict & Explain** | Input subject data → get prediction + SHAP waterfall/force/breakdown |
| 📊 **Model Evaluation** | AUC, Accuracy, Precision, Recall, F1, Confusion Matrix, ROC Curve |
| 🌐 **Global SHAP** | Feature importance ranking, Bar plot, Beeswarm plot, Dependence plot |
| 📋 **Validation Samples** | Browse any validation sample with SHAP explanation |

---

## 🚀 Deployment Guide (GitHub → Streamlit Cloud)

### Step 1: Set Up Git Locally
```bash
cd D:\workBuddy\shap-eyemove-app
git init
git add .
git commit -m "Initial commit: Eye Movement SHAP Calculator"
```

### Step 2: Create GitHub Repository
1. Go to **https://github.com** → log in
2. Click **"+"** (top right) → **"New repository"**
3. Name: `shap-eyemove-app`
4. Visibility: **Public** ✅ (required for Streamlit Cloud free tier)
5. Do NOT initialize with README
6. Click **"Create repository"**

### Step 3: Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/shap-eyemove-app.git
git branch -M main
git push -u origin main
```

### Step 4: Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io** → sign in with GitHub
2. Click **"New app"**
3. Fill in:
   - Repository: `YOUR_USERNAME/shap-eyemove-app`
   - Branch: `main`
   - Main file: `app.py`
4. Click **"Deploy!"**
5. Wait ~2-3 minutes → app goes live

### Step 5: Auto-Update
Every push to `main` triggers auto-redeploy:
```bash
git add .
git commit -m "Update description"
git push
```

---

## 🖥️ Run Locally

```bash
cd D:\workBuddy\shap-eyemove-app
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🛠️ Tech Stack
- **Streamlit** 1.56+ — Web framework
- **scikit-learn** 1.8+ — Random Forest classifier
- **SHAP** 0.44+ — Model explanation
- **Matplotlib / Seaborn** — Visualizations
- **Pandas / NumPy** — Data processing
