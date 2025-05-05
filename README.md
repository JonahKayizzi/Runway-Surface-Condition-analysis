
# 🛬 Runway Surface Condition Monitoring and GRF Reporting

This project implements a deep learning system for **runway surface condition classification**, **trend analysis**, and **forecasting** using image sequences. It generates **ICAO-compliant GRF (Runway Condition Code)** reports based on classified observations and weather data.

---

## 📌 Project Overview

The system integrates:
- ✅ **2D CNN (EfficientNet-B3)** for classifying individual runway frames into ICAO GRF codes
- ✅ **3D CNN (ResNet3D)** for analyzing temporal trends from image sequences
- ✅ **Regression model** for forecasting future GRF conditions using visual + weather data
- ✅ **ICAO-compliant report generation** in JSON and SNOWTAM formats

---

## 🧠 Core Features

| Component             | Description |
|----------------------|-------------|
| **2D Classifier**     | EfficientNet-B3 trained on RoadSaW + RoadSC datasets |
| **3D CNN**            | ResNet3D (`r3d_18`) detects surface condition trends |
| **Trend Analysis**    | Uses sliding window of 3D outputs to label trend as **Improving**, **Worsening**, or **Stable** |
| **Future Regressor**  | Predicts future GRF code using extracted features + weather forecast |
| **ICAO Reporting**    | Outputs structured runway condition reports in ICAO GRF and SNOWTAM formats |
| **Visualization**     | Trend plots and annotated frame previews |


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/gilbert215/Runway-weather-analysis.git
cd Runway-weather-analysis
```


## ✈️ ICAO GRF Classes

| RWYCC | Description              |
|-------|--------------------------|
| 6     | DRY                      |
| 5     | WET (≤3mm water)         |
| 4     | COMPACTED SNOW           |
| 3     | SLIPPERY WET             |
| 2     | STANDING WATER (>3mm)    |
| 1     | ICE                      |
| 0     | WET ICE (slush on ice)   |

The GRF labels are mapped from road surface datasets using film height, snow coverage, and surface descriptors.

---

## 🌦️ Weather Forecast Integration

- Uses [OpenWeatherMap API](https://openweathermap.org/forecast5)
- Fetches **30-minute forecast** based on time and location (Kigali, default)
- Inputs used:
  - Temperature (°C)
  - Precipitation description
  - Cloud coverage (%)
- Used to improve future GRF prediction via regression

---

## 📈 Outputs

- ✅ Per-frame GRF classifications (0–6)
- ✅ Trend labels across image sequences
- ✅ GRF forecast for the next 30 minutes
- ✅ ICAO-formatted reports in:
  - `JSON`
  - `SNOWTAM` textual format
- ✅ Trend plots and GRF evolution charts

---

## 📄 License and Citation


## 🙌 Acknowledgments

- [RoadSaW Dataset](https://www.viscoda.com/index.php/downloads/roadsaw-dataset)
- [RoadSC Dataset](https://www.viscoda.com/index.php/downloads/roadsc-dataset)
- [OpenWeatherMap API](https://openweathermap.org/api)
- ICAO GRF Reporting Framework

---
