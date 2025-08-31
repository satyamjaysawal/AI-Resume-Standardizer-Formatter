












# 🌟 GenAI-CV-ML-CaseStudies

This repository contains my submission for the **Senior AI Engineer Technical Assignment (Round 2)**.
It includes **three real-world AI/ML projects** across **Generative AI, Computer Vision, and Machine Learning** domains.

👉 **One project is fully implemented in code**
👉 **Two projects are submitted as architecture/approach documents**

---

## 📂 Repository Contents

| Project                                              | Type                  | File / Link                                                                                                                                               |
| ---------------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Resume Standardization (GenAI Use Case)**          | ✅ Full Implementation | [AI-Driven-Resume-Formatter.md](./AI-Driven-Resume-Formatter.md) <br> + Supporting code (`main.py`, `requirements.txt`, `coords.json`, `template/`, etc.) |
| **Ice Coverage Checking (Computer Vision Use Case)** | 📄 Architecture Doc   | [Ice-Coverage-Checking-Project.md](./Ice-Coverage-Checking-Project.md)                                                                                    |
| **Sales Forecasting (Traditional ML Use Case)**      | 📄 Architecture Doc   | [Sales-Forecasting–Traditional-ML-UseCase.md](./Sales-Forecasting–Traditional-ML-UseCase.md)                                                              |

---

## ✅ Fully Implemented: Resume Standardization (GenAI)

**Objective**: Build a **Generative AI application** that takes resumes in any format (PDF/DOCX) and converts them into a **standardized DOCX resume template**.

**Highlights**:

* Accepts resumes in **PDF/DOCX** format.
* Extracts information (name, skills, education, experience, etc.) using **Azure Document Intelligence**.
* Structures & rephrases content into **professional tone** using **Azure OpenAI GPT-4o**.
* Populates data into a **predefined DOCX template** using precise coordinate mapping.
* Outputs a **professionally formatted, editable .docx resume**.

**Bonus Features**:

* Supports external tone/style instructions (e.g., formal, concise).
* Error handling for misaligned sections or content overflow.

---

### 🔹 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

**Inputs:**

* `input/` → Resume files (`.pdf` or `.docx`)
* `template/` → Standard resume template (`.docx`)
* `coords.json` → Coordinate mapping for fields
* `.env` → Azure service credentials

**Outputs:**

* `output/filled-resume.docx` → Standardized formatted resume

---

## 📝 Architecture/Approach Documents

### 1. Ice Coverage Checking – Computer Vision

* Detect crates in video using **YOLOv8 + SORT tracking**.
* Crop crate top → estimate % ice coverage with **Azure OpenAI Vision**.
* Trigger alerts if coverage < threshold (e.g., 80%).
* Bonus: Worker safety compliance detection (coat, gloves, hat).

### 2. Sales Forecasting – Traditional ML

* Aggregate weekly → monthly sales data.
* Perform **EDA** (seasonality, anomalies, customer/plant patterns).
* Train **XGBoost/Prophet models** for forecasting.
* Forecast next 12 months (amount & quantity) at both **customer & plant level**.
* Bonus: Handle new/churned/irregular customers.

---

## ⚡ Challenges & Future Enhancements

* **Resume Formatter**: Automating template coordinate mapping; adding web-based upload/preview UI.
* **Ice Coverage**: Stability across lighting conditions; real-time performance improvements.
* **Sales Forecasting**: Hierarchical forecasting; improving accuracy for sparse customer segments.

---

## 🏆 Closing Notes

This submission demonstrates:

* **GenAI application development** (resume formatter).
* **Computer Vision pipeline design** (ice coverage).
* **Traditional ML forecasting architecture** (sales forecasting).

---































