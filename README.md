# Flight Accuracy Prediction

A machine learning project for predicting flight delays and accuracy using weather data and airline information across multiple countries.

## Overview

This project develops predictive models to forecast flight delays and on-time performance by analyzing historical flight data combined with weather patterns from 30+ countries. The system uses advanced machine learning algorithms to provide accurate predictions for airline operations and travel planning.

## Features

- **Multi-Country Weather Integration**: Incorporates weather data from 30+ countries including USA, UK, China, India, Brazil, and more
- **Comprehensive Flight Data Analysis**: Analyzes airport frequencies, airline information, and flight patterns
- **Multiple ML Models**: Implements various machine learning algorithms for comparison and ensemble methods
- **Performance Visualization**: Includes confusion matrices, ROC curves, calibration charts, and feature importance plots
- **Web Interface**: Flask-based web application for easy interaction and predictions
- **Detailed Reporting**: Generates comprehensive evaluation reports with model comparisons

## Project Structure

```
flight-accuracy/
├── app.py                          # Flask web application
├── main.py                         # Main entry point
├── train_and_save.py              # Model training and saving
├── generate_report.py             # Report generation
├── airlines.json                  # Airline data
├── airports.json                  # Airport information
├── flight/                        # Flight data and weather datasets
│   ├── weather_data_top100.csv
│   ├── airports.csv
│   ├── airport-frequencies.csv
│   └── [Country]_weather_data.csv (30+ weather files)
├── flight_data/                   # Processed flight data
├── models/                        # Trained ML models
├── static/                        # Static assets (CSS, JavaScript)
│   └── css/
│   └── js/
├── templates/                     # HTML templates
├── paper_images/                  # Research paper images
├── paper_full.txt                # Full research paper
└── detailed_evaluation_report.txt # Model evaluation results
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/hariprasad7072/Flight-Accuaracy-Prediction.git
   cd Flight-Accuaracy-Prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Models

Train and save ML models using the available flight and weather data:

```bash
python train_and_save.py
```

### Running the Web Application

Start the Flask web application:

```bash
python app.py
```

Access the application at `http://localhost:5000`

### Generating Reports

Generate comprehensive evaluation reports:

```bash
python generate_report.py
```

### Main Analysis

Run the main analysis pipeline:

```bash
python main.py
```

## Data Sources

- **Weather Data**: 30+ countries with comprehensive climate information
- **Airline Data**: Multiple airlines with operational details
- **Airport Data**: Airport frequencies and geographic information
- **Flight Data**: Historical flight records with delay information

## Model Performance

The project implements and compares multiple machine learning models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machines
- Neural Networks

Key evaluation metrics:
- Accuracy
- Precision and Recall
- ROC-AUC Score
- F1 Score
- Calibration Analysis

## Visualizations

The project generates comprehensive visualizations:

- **Confusion Matrices**: Multi-model comparison
- **ROC Curves**: Model discrimination ability
- **Learning Curves**: Model convergence analysis
- **Calibration Curves**: Probability calibration
- **Feature Importance**: Top predictive features
- **Lift Charts**: Model lift analysis
- **Precision-Recall Curves**: Performance metrics
- **Performance Radar**: Multi-metric comparison

## Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **XGBoost & LightGBM**: Gradient boosting frameworks
- **TensorFlow/Keras**: Neural networks
- **Flask**: Web framework
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations

## Results

The models achieve high prediction accuracy with comprehensive analysis of:
- Feature importance across different algorithms
- Cross-validation analysis
- Hyperparameter tuning effects
- Ensemble method performance

Detailed results are available in `detailed_evaluation_report.txt`

## Research

Complete research documentation available in:
- `paper_full.txt` - Full research paper
- `paper_text.txt` - Paper text content
- `paper_images/` - Research figures and diagrams

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is available for educational and research purposes.

## Contact

For questions or inquiries, please reach out through GitHub issues or contact the project maintainer.

## Acknowledgments

- Weather data providers for comprehensive climate information
- Airline and airport data sources
- Machine learning community for algorithm implementations
- Contributors and testers

---

**Last Updated**: April 2026
**Project Status**: Active Development
