# FX Forecaster: An Exchange Rate Predication App

**A smart web application that forecasts 30-day foreign exchange rates using Deep Learning.**
Built with **Django**, **TensorFlow**, and **Plotly**, this project demonstrates a full end-to-end Machine Learning + Web App integration from model design to dynamic UI. 

---

## Application Preview

Here are the two main modes of the application:

### Live Training Forecast (Default)
### Instant Forecast (Pre-Trained)

<table style="width:100%;">
  <thead>
    <tr>
      <th style="text-align:left;">Mode</th>
      <th style="text-align:left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Live Training Forecast (Default)</b></td>
      <td>Builds a new Tensorflow model using live currency data for real-time forecasting.</td>
    </tr>
    <tr>
      <td><b>Instant Forecast (Pre-Trained)</b></td>
      <td>Loads pre-trained models for near-instant predictions.</td>
    </tr>
  </tbody>
</table>

---

## Project Overview

This project was born from a real-world problem I faced. While applying to university in the UK, I and many friends were trying to plan how we all were going to manage student loans and tuition fee payments from abroad.  
We were constantly stressing about the fluctuating exchange rates (like GBP/INR) and trying to guess the best time to transfer money.

Everyone kept asking, "What will the rate be next few days and the next month?" But we couldn't find a simple tool to provide a reasonable estimate.  
So, I decided to build one. This project is the result: a tool that attempts to answer that question, built from the ground up as a practical application of my data science skills.

---

## Key Features

- **Dual Forecast Modes**
  - Live Training: Retrains an TensorFlow model in real-time using the most recent data.
  - Instant Forecast: Loads pre-trained models for rapid predictions.
- **Any-to-Any Cross-Rate System**
  - Predicts exchange rates between *any two currencies* by combining USD-based models.
- **Interactive Visualization**
  - Built with Plotly to provide interactive, zoomable, and hover-enabled forecast charts.
- **Deep Learning Architecture**
  - Optimized TensorFlow models trained with EarlyStopping for accuracy and stability.
- **Clean Django Front-End**
  - A dual-page interface (Live vs. Instant) with clear routing and dynamic chart updates.

---

## Conceptual Architecture

```
User → Django View → Data Fetch (European Central Bank)
        ↓
    Model (Pre-Trained or Live-Trained)
        ↓
     Forecast Output → Plotly Visualization → Browser
```

### Instant Forecast (Fast)
1. Loads two pre-trained models: `CurrencyA_USD.h5` and `CurrencyB_USD.h5`
2. Generates 30-day predictions for each
3. Computes cross-rate forecast:  
   **Forecast(A/B) = Forecast(A/USD) / Forecast(B/USD)**
4. Displays results instantly

### Live Training (Slow)
1. Fetches the latest currency pair data via **European Central Bank**
2. Builds and trains a new model (~30 - 80 epochs)
3. Produces a fresh 30-day forecast
4. Displays interactive chart results

---

## Tech Stack

| Layer | Tools |
|-------|--------|
| **Backend** | Django, Python 3.11 |
| **ML / AI** | TensorFlow, Keras |
| **Data Science** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Plotly, Matplotlib |
| **Data Source** | European Central Bank |
| **Storage** | HDF5 (.h5), Joblib (.joblib) |
| **Frontend** | HTML5, CSS3, JavaScript |

---

## Repository Structure (Public Version)

```
fx-forecaster/
├── .gitignore             
├── README.md              
├── requirements.txt        # List of Python packages
├── sample_env.txt          
├── manage.py               # Django utility
│
├── fx_project/             # Django project configuration
│   ├── settings.py         
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── forecaster/             # Django 
│   ├── apps.py
│   ├── models.py
│   ├── tests.py          
│   └── urls.py
│
└── templates/              # HTML templates
  └── forecaster/
      ├── index.html
      ├── fx.html
      └── instant_fx.html
```

---

## Private Components (Not Included in Public Repo)

To protect intellectual property and API security, the following files are intentionally excluded:

- `.env` file (contains API key)
- Models irectory (trained model weights & scalers)
- `models.py` (offline model training logic)
- `utils.py` (data preprocessing & model architecture)
- Real-time forecasting logic inside `views.py`
- Database and cached files

---

## License & Usage

```
© 2025 Manoj Chandra
All rights reserved.

This repository is provided for educational and demonstration purposes only.
Redistribution, commercial use, or reuse of core model logic, weights, or datasets
is prohibited without explicit permission from the author.
```

---

## Author

**Manoj Chandra** [LinkedIn](https://www.linkedin.com/in/your-profile)  [GitHub](https://github.com/your-username)

---

## Acknowledgements

- [European Central Bank](https://data.ecb.europa.eu/) – Currency Data  
- [TensorFlow](https://www.tensorflow.org/) – Deep Learning Framework  
- [Plotly](https://plotly.com/python/) – Data Visualization  

---

## Final Note

FX Forecaster is a real working project demonstrating:
- **Practical ML deployment** in a web app  
- **Data-driven problem solving**, and  
- **End-to-end architecture design**

If you’d like to collaborate or discuss similar projects — feel free to reach out! 💬
