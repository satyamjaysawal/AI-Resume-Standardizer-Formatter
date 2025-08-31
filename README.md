




# 🌟 GenAI-CV-ML-CaseStudies

This repository contains my submission for the **Senior AI Engineer Technical Assignment (Round 2)**.
It includes **three real-world AI/ML projects** across **Generative AI, Computer Vision, and Machine Learning** domains.

👉 **One project is fully implemented in code**
👉 **Two projects are submitted as architecture/approach documents**

---

## 📂 Repository Contents

| Project                                              | Type                  | File / Link                                                                                                                                                                                                                  |
| ---------------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Resume Standardization (GenAI Use Case)**          | 📄 Architecture Doc   | [AI-Driven-Resume-Formatter.md](https://github.com/satyamjaysawal/AI-Resume-Standardizer-Formatter/blob/main/AI-Driven-Resume-Formatter.md)                                                                                  |
| **Ice Coverage Checking (Computer Vision Use Case)** | 📄 Architecture Doc   | [Ice-Coverage-Checking-Project.md](https://github.com/satyamjaysawal/AI-Resume-Standardizer-Formatter/blob/main/Ice-Coverage-Checking-Project.md)                                                                            |
| **Sales Forecasting (Traditional ML Use Case)**      | ✅ Full Implementation | [Sales-Forecasting–Traditional-ML-UseCase.md](https://github.com/satyamjaysawal/AI-Resume-Standardizer-Formatter/blob/main/Sales-Forecasting%E2%80%93Traditional-ML-UseCase.md) <br> + Code in `main.py`, `requirements.txt` |

---

## ✅ Implemented in Full: Sales Forecasting (Traditional ML)

**Objective**: Forecast sales **amount (\$)** and **quantity (lbs.)** for the next 12 months (Oct 2025 – Sep 2026), at both **customer** and **plant** levels.

**Highlights**:

* Preprocessed weekly → monthly aggregated sales data.
* Conducted **EDA** (seasonality, anomalies, plant-level trends).
* Engineered features (lags, rolling means, Fourier terms).
* Trained **XGBoost models** for regression.
* Evaluated with **MAPE, RMSE**, validated **≥80% accuracy at plant level**.
* Forecasts and plots saved in `output/`.

### 🔹 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess input data (train + test → monthly format)
python preprocess.py

# Run forecasting pipeline
python main.py
```

**Outputs:**

* `output/forecasts.csv` → forecasted sales for Oct 2025 – Sep 2026
* `output/plots/` → visualizations of trends & forecasts

---

## 📝 Approach Documents

### 1. Resume Standardization – GenAI

* Extracts resumes in any format (PDF/DOCX).
* Uses **Azure Document Intelligence + OpenAI GPT-4o** for structured extraction & rephrasing.
* Populates into standardized **DOCX template**.
* Bonus: Supports tone/style guidelines.

### 2. Ice Coverage Checking – Computer Vision

* Detects shrimp crates in surveillance video.
* Crops crate top → estimates % ice coverage via **Azure OpenAI Vision**.
* Flags crates below threshold (e.g., <80% coverage).
* Bonus: Worker safety compliance (coat, gloves, hat) detection.

---

## ⚡ Challenges & Future Enhancements

* **Sales Forecasting**: Handle sparse/irregular customers via hierarchical models.
* **Resume Formatter**: Automate Word template coordinate mapping, add web UI.
* **Ice Coverage**: Improve robustness across lighting, optimize Azure API usage.

---

## 🏆 Closing Notes

This submission demonstrates:

* **GenAI application design** (resume standardization).
* **Computer Vision + Vision-Language integration** (ice coverage).
* **Traditional ML for forecasting** (sales pipeline).

---
































