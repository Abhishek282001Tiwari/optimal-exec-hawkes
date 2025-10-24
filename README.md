# Optimal Execution under Self-Exciting Order Flow: A Stochastic Control Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Abstract

This research presents a novel optimal execution framework that combines Hawkes processes with linear-quadratic (LQ) stochastic control to address self-exciting order flow in financial markets. The methodology captures clustering effects in market microstructure and provides closed-form solutions for optimal execution under realistic market conditions.

## 🎯 Key Contributions

- **Novel Framework**: First application of Hawkes-LQ control to optimal execution
- **Empirical Validation**: Calibrated on real market data from 5 major stocks
- **Performance Improvement**: 8-15% cost reduction vs traditional strategies
- **Practical Implementation**: Complete codebase for replication and extension

## 📊 Results Summary

| Metric | Improvement |
|--------|-------------|
| Cost Reduction vs AC | 8-15% |
| Risk Reduction (VaR) | Significant |
| Market Regimes Tested | 3×3 grid |
| Real Data Validation | 5 stocks, 1,255 points |

## 🏗️ Project Structure
optimal-exec-hawkes/
├── notebooks/ # Jupyter notebooks for experiments
├── scripts/ # Python scripts for analysis
├── calib/ # Calibration modules
├── sim/ # Simulation modules
├── backtest/ # Backtesting framework
├── models/ # Mathematical models
├── docs/ # Documentation and figures
└── data/ # Market data (git-ignored)


## 🚀 Quick Start

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/optimal-exec-hawkes.git
   cd optimal-exec-hawkes

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```bash
3. **Run Experiments**
# Run the complete experimental pipeline
python scripts/run_experiments.py

4. **Generate Paper Figures**
python scripts/paper_plots/essential_figures.py

**Key Features**

Methodological

Hawkes process calibration on real market data
Stochastic control with self-exciting dynamics
Endogenous feedback modeling
Multi-asset framework
Empirical

Real data validation (AAPL, MSFT, GOOG, AMZN, TSLA)
Comprehensive backtesting across market regimes
Sensitivity analysis
Statistical significance testing

##Citation

If you use this code in your research, please cite:

@article{yourpaper2024,
  title={Optimal Execution under Self-Exciting Order Flow: A Stochastic Control Framework},
  author={Abhishek Tiwari},
  journal={Working Paper},
  year={2025}
}

**Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

**License**

This project is licensed under the MIT License - see the LICENSE file for details.


### **Step 3: Create Requirements.txt**
```bash
# Create requirements file
cat > requirements.txt << 'EOF'
# Core Scientific Computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Machine Learning & Statistics
scikit-learn>=1.0.0
statsmodels>=0.13.0

# Data Acquisition
yfinance>=0.2.0
requests>=2.25.0

# Progress Bars
tqdm>=4.62.0

# Jupyter for Notebooks
jupyter>=1.0.0
ipykernel>=6.0.0

# Optional: Advanced Optimization
cvxpy>=1.1.0

# Development
black>=22.0.0
flake8>=4.0.0
pytest>=6.0.0
EOF

