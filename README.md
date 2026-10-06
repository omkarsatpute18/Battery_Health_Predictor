# 🔋 Battery Health Predictor

A Machine Learning project for predicting **battery capacity** using battery operating and performance parameters such as voltage, current, temperature, charging time, discharging time, internal resistance, and battery cycle.

## 📌 Project Overview

Battery performance gradually changes as the number of charge-discharge cycles increases. Monitoring battery health can help understand battery degradation and estimate its remaining performance.

This project uses historical battery-cycle data and Machine Learning regression techniques to predict **battery capacity (Ah)** from measurable battery parameters.

The project was developed using Python and Jupyter Notebook.

## 🎯 Objective

The main objective of this project is to:

- Analyze battery performance data.
- Clean and preprocess the dataset.
- Handle missing values.
- Explore relationships between battery parameters.
- Build Machine Learning regression models.
- Predict battery capacity.
- Evaluate model performance using regression metrics.

## 📊 Dataset

The dataset contains **10,000 records** and initially contains **10 features**.

### Features

| Feature | Description |
|---|---|
| `battery_id` | Unique identifier of the battery |
| `cycle` | Battery charge/discharge cycle number |
| `avg_voltage_V` | Average battery voltage in volts |
| `avg_current_A` | Average current in amperes |
| `temperature_C` | Battery temperature in °C |
| `charge_time_hr` | Charging time in hours |
| `discharge_time_hr` | Discharging time in hours |
| `internal_resistance_ohm` | Internal resistance of the battery in ohms |
| `capacity_Ah` | Battery capacity in ampere-hours |
| `soh_percent` | State of Health of the battery in percentage |

The dataset contains missing values in several sensor-related columns, which are handled during preprocessing.

## 🔎 Exploratory Data Analysis

The notebook performs exploratory analysis including:

- Dataset inspection
- Dataset shape and information
- Statistical summary
- Missing-value analysis
- Battery cycle analysis
- Battery capacity analysis
- Visualization of relationships between battery parameters

The dataset contains 10,000 rows, with battery IDs ranging from 1 to 40 and cycles ranging from 1 to 250.

## 🧹 Data Preprocessing

Missing values are handled using **mean imputation**.

The following columns contain missing values and are filled using their respective column means:

- `avg_voltage_V`
- `avg_current_A`
- `temperature_C`
- `charge_time_hr`
- `discharge_time_hr`
- `internal_resistance_ohm`

After preprocessing, the dataset contains no missing values.

The `battery_id` column is removed before model development because it acts as an identifier rather than a useful predictive feature. This leaves **9 columns** for the modeling dataset.

## 🤖 Machine Learning

The project uses regression algorithms to predict battery capacity.

### Linear Regression

A Linear Regression model is trained using standardized input features.

The project uses `StandardScaler` to standardize the training and testing features before applying Linear Regression. 

### Random Forest Regression

A Random Forest Regressor is also implemented with:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

The model is trained on the battery dataset and used to generate capacity predictions.

## 📈 Model Evaluation

The Linear Regression model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Linear Regression Results

| Metric | Result |
|---|---:|
| MAE | 0.0734 Ah |
| RMSE | 0.0865 Ah |
| R² Score | 0.2731 |

The notebook also compares actual and predicted battery capacity values to evaluate the model's prediction behavior.

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 📂 Project Structure

```text
Battery_Health_Predictor/
│
├── main.ipynb
├── battery_health_data.csv
└── README.md
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/omkarsatpute18/Battery_Health_Predictor.git
```

### 2. Navigate to the project

```bash
cd Battery_Health_Predictor
```

### 3. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
main.ipynb
```

Make sure `battery_health_data.csv` is present in the same project directory.

## 🚀 Future Improvements

Possible improvements for this project include:

- Hyperparameter tuning of the regression models.
- Comparing additional regression algorithms.
- Improving prediction accuracy.
- Feature engineering based on battery degradation characteristics.
- Deploying the trained model as a web application.
- Adding real-time battery health prediction.
- Developing a dashboard for battery monitoring.

## 👨‍💻 Author

**Omkar Satpute**

GitHub: [@omkarsatpute18](https://github.com/omkarsatpute18)

---

⭐ If you find this project useful, consider giving the repository a star!