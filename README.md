# Survey-Analysis

A comprehensive Python pipeline designed to process, clean, analyze, and visualize complex survey response data. This repository provides end-to-end data analysis workflows—from handling missing data and text processing to generating statistical summaries and presentation-ready visualizations.

## 🚀 Key Features

- **Automated Data Cleaning**: Robust pipelines to handle missing values, encode categorical variables, filter outliers, and clean textual survey descriptions.
- **Statistical Summaries**: Cross-tabulations, demographic deep-dives, and descriptive stats mapping key metrics.
- **Advanced Visualizations**: Beautiful, publication-grade distribution plots, sentiment analysis maps, and multi-variable comparison charts.
- **Insight Extraction**: Sentiment distribution markers or key behavioral segmentation drivers behind survey responses.

## 📁 Repository Structure

```text
├── data/                  # Raw and processed survey data (.csv, .xlsx)
│   ├── raw/               # Immutable raw survey responses
│   └── processed/         # Cleaned and engineered data ready for analysis
├── notebooks/             # Jupyter notebooks for interactive analysis and EDA
│   └── exploratory_analysis.ipynb
├── src/                   # Source code modules for reuse
│   ├── __init__.py
│   ├── cleaning.py        # Scripts for handling nulls, type-casting, and encoding
│   ├── analysis.py        # Statistical computations and aggregation functions
│   └── plotting.py        # Custom visualization functions (matplotlib/seaborn)
├── .gitignore             # Ensures raw data and environment files aren't tracked
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation
```

## 🛠️ Installation & Setup

Make sure you have Python 3.8+ installed on your system. 

### 1. Clone the Repository
```bash
git clone https://github.com/kanishk-001/Survey-Analysis.git
cd Survey-Analysis
```

### 2. Set Up a Virtual Environment (Recommended)
Using Python's built-in `venv`:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Alternatively, if you use `uv`:
```bash
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Core Dependencies
This pipeline is powered by the standard scientific Python stack:
- **Pandas** & **NumPy** for data frame manipulation and array calculations.
- **Matplotlib** & **Seaborn** for exploratory visualization profiles.
- **Scikit-Learn** or **NLTK** *(if applicable for text/clustering workflows)*.

## 🏃‍♂️ How to Run the Analysis

### Running via Jupyter Notebooks
Open the notebooks directory to interactively step through the data pipeline and exploratory visual elements:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Running via Scripts
If you prefer execution directly from your terminal interface:
```bash
python src/cleaning.py
python src/analysis.py
```

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
