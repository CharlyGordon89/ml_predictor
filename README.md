# ml_predictor

**Reusable prediction module for loading ML models and generating predictions in production pipelines.**

---

## 🔧 Purpose

This module provides a clean interface for loading serialized models and making predictions from structured input (e.g., JSON, dicts, or DataFrames). It is designed to be **portable**, **testable**, and **reusable** across any ML project — whether local, on-premise, or deployed to cloud platforms like AWS/GCP/Azure.

---

## ✅ Features

- 📦 Load any serialized model (`joblib`, `pickle`)
- 🔮 Predict using structured input (dict, DataFrame, or JSON)
- ✅ Handles input parsing and model inference
- 🧪 Comes with full unit tests and dummy input
- 🔁 Reusable across projects (classification, regression, etc.)

---

## 📁 Project Structure

