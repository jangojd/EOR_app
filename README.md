# EOR Method Prediction App

A machine learning-powered web application built with Streamlit for predicting Enhanced Oil Recovery (EOR) methods based on reservoir characteristics.

## Overview

This application uses a trained Random Forest model to predict the most suitable EOR method for oil reservoirs based on various geological and fluid properties. The app provides probability distributions for different EOR methods, helping petroleum engineers make informed decisions.

## Features

- **Interactive Input Interface**: User-friendly sidebar for entering reservoir parameters
- **Real-time Predictions**: Instant EOR method recommendations
- **Probability Visualization**: Bar chart showing probabilities for all EOR methods
- **Detailed Results**: Tabular display of prediction probabilities

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Libraries

Install the required dependencies using pip:

```bash
pip install pandas
pip install altair
pip install numpy
pip install streamlit
pip install scikit-learn
```

Or install all at once:

```bash
pip install pandas altair numpy streamlit scikit-learn
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jangojd/jangojd
   cd jangojd
   ```

2. **Ensure the trained model file exists**:
   - The app requires `trained_model.sav` in the same directory
   - This file contains the pre-trained Random Forest model

3. **Run the application**:
   ```bash
   streamlit run test7.py
   ```

4. **Access the app**:
   - Open your browser and navigate to `http://localhost:8501`

## Input Parameters

The application accepts the following reservoir and fluid properties:

| Parameter | Description | Unit | Range |
|-----------|-------------|------|-------|
| Area | Reservoir area | acres | 0 - 100,000 |
| Porosity | Rock porosity | % | 0 - 100 |
| Permeability | Rock permeability | md | 0 - 100,000 |
| Depth | Reservoir depth | ft | 0 - 1,000,000 |
| Gravity | Oil gravity | API | 0 - 10,000 |
| Viscosity | Oil viscosity | cp | 0 - 100,000 |
| Temperature | Reservoir temperature | °F | 0 - 30,000 |
| Saturation | Oil saturation | % | 0 - 100 |
| Formation | Geological formation | - | Multiple options |

### Supported Formations

S, Dolo, Carb, Sh, LS, US, Tripol, Cong, LS or Dolo, DoloorTripol, LSorDolo, Dolo or S, SS, Congl, SorLS-Dolo

## Output

The application provides:

1. **Primary Prediction**: The most likely EOR method for the given parameters
2. **Probability Bar Chart**: Visual representation of all EOR method probabilities
3. **Probability Table**: Detailed breakdown of prediction confidence for each method

## Screenshots

### Main Interface
![App Interface](https://github.com/jangojd/EOR_app/blob/a78462b5b316399a9c146a641b7d3ee509de058a/Screenshot%20(9).png)


> 

## Project Structure

```
.
├── test7.py              # Main application file
├── trained_model.sav     # Pre-trained Random Forest model
├── README.md             # Project documentation
└── screenshots/          # Application screenshots
    ├── app_interface.png
    └── prediction_results.png
```

## Model Information

- **Algorithm**: Random Forest Classifier
- **Model File**: `trained_model.sav` (pickle format)
- **Features**: 9 input parameters (8 numerical + 1 categorical)
- **Output**: Multi-class classification of EOR methods

## Authors

**Main Author:**
- Jawad Ali

**Co-Authors:**
- Ali Akbar
- Ali Zaheer

## Repository

GitHub: [https://github.com/jangojd/jangojd](https://github.com/jangojd/jangojd)

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


---
