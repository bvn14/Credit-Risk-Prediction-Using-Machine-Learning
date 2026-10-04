# Credit Risk Prediction Using Machine Learning

**Course:** DA4641 Introduction to FinTech, Case Study 01 (Python-Based FinTech Analytics)
**Group:** 6, Dataset 1
**Institution:** Department of Decision Sciences, Faculty of Business, University of Moratuwa

## Team

| Name | Index No. |
|---|---|
| Kumara S.D.N.S | 226067E |
| Weerasekara R.D | 226133E |
| Gunasinghe P.B | 226044G |
| Jayathissa W.A.G.J.U.N | 216057N |
| Ranaweera R.P.N.L | 216104H |

## Project Overview

This project builds and evaluates a machine learning model to predict loan default for a lending institution. The target variable is `default_status` (1 = default, 0 = non-default). The aim is not only good prediction, but also a model that is interpretable, fair to review, and useful in real lending decisions.

The work covers data cleaning, exploratory analysis, model comparison, threshold selection using business cost, interpretation, fairness checks, and a discussion of limitations.

## Dataset

- File: `dataset01.csv`
- Original size: 4,715 rows and 23 columns
- After removing 115 duplicate rows: 4,600 unique applicants
- Default rate: about 28.3%
- Main fields: age, income, debt payments, debt-to-income ratio, credit score, delinquencies, employment, loan details, previous default, region and application date

Data quality problems found and handled: duplicates, missing values, inconsistent text labels, three different date formats, and impossible values (for example age above 100, employment length of 999, credit scores outside 300 to 850).

## Methodology

1. **Cleaning:** Rule-based fixes only. Impossible values were set to missing. Labels were standardised.
2. **Feature selection:** `customer_id` and `application_date` were not used for prediction. `gender` and `marital_status` were excluded from the model and kept only for fairness analysis. `interest_rate_pct` was removed because it has almost no link to default and may be set by the lender after a risk assessment (circular logic). 18 predictors were considered at the start.
3. **Split:** 80:20 stratified train-test split (3,680 train, 920 test).
4. **Pre-processing pipeline:** Imputation, scaling and one-hot encoding are done inside a scikit-learn pipeline after the split, to avoid data leakage.
5. **Models compared:** Logistic Regression, Random Forest, Gradient Boosting, plus a dummy baseline. All were checked with 5-fold stratified cross-validation and light hyperparameter tuning.
6. **Model choice:** Logistic Regression was chosen because its cross-validated ROC-AUC was within 0.005 of the best model, and it is simpler and easier to explain. Class weighting was added to improve recall on defaulters.
7. **Threshold:** Chosen at 0.41 using out-of-fold predictions and an assumed 3:1 cost for a missed defaulter versus a wrongly flagged applicant.

## Key Results

| Item | Result |
|---|---|
| Dummy baseline | About 72% accuracy, ROC-AUC 0.500, recall 0 |
| Cross-validated ROC-AUC after tuning | Logistic Regression 0.687, Random Forest 0.687, Gradient Boosting 0.691 |
| Final test ROC-AUC | 0.715 (bootstrap 95% CI about 0.678 to 0.750) |
| Gini coefficient | 0.430 |
| PR-AUC | About 0.463 |
| Recall at threshold 0.41 | About 83.5% |
| Precision at threshold 0.41 | About 39.1% |
| False-positive rate | About 51.2% |
| Applicants flagged high risk | About 60.3% |
| Out-of-time test (train 2021 to 2023, test 2024) | ROC-AUC about 0.693 |

**Main drivers of default:** credit score (strongest), previous default, unemployment and higher debt-to-income ratio.

**Risk bands:** The observed default rate rises from about 9.2% in the lowest-risk group to 50.0% in the highest-risk group.

## Practical Implications

The model catches most defaulters, but it also flags many good applicants. It should be used as a **risk screening and ranking tool** with human review, not as an automatic rejection system. Example use: normal process for low risk, extra checks for medium risk, detailed review for high risk.

## Limitations and Risk Considerations

- The data looks synthetic or simplified. The 28.3% default rate may not match a real portfolio, and the default definition is not documented.
- Some records contain contradictory values (for example age and employment length), and the date formats were interpreted without clear documentation.
- Performance drops slightly on the out-of-time test, so results may change over time.
- Removing gender and marital status does not remove all bias, because other variables can act as proxies. Some small groups gave unstable estimates.
- The 3:1 cost ratio is an assumption. A real institution should use actual loss, recovery and profit data.
- Because of class weighting, predicted probabilities are not true default probabilities. Calibration is needed if they are used that way.

## Future Work

- Test on larger, real-world data and longer out-of-time periods.
- Calibrate predicted probabilities.
- Set the threshold using real business costs.
- Monitor performance, data drift and fairness after deployment.

## Repository Structure

```
.
├── README.md
├── dataset01.csv
├── Group_06_DA4641_Dataset_01_Python.ipynb
└── CA1_fintech_group_6.pdf
```

## How to Run

**Option 1: Google Colab**
1. Upload the notebook to Colab.
2. Run the cells in order. When asked, upload `dataset01.csv`.

**Option 2: Local machine**
1. Clone the repository and open the folder.
2. Install the requirements:
```
   pip install -r requirements.txt
```
3. Make sure `dataset01.csv` is in the same folder as the notebook.
4. Start Jupyter and run all cells in order:
```
   jupyter notebook
```

The notebook detects automatically if it is running in Colab or on a local computer.
**Option 1: Google Colab**
1. Upload the notebook to Colab.
2. Run the cells in order. When asked, upload `dataset01.csv`.

**Option 2: Local machine**
1. Install the requirements:
```
   pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
```
2. Put `dataset01.csv` in the same folder as the notebook.
3. In the notebook, remove or skip the cell that uses `from google.colab import files`, and load the CSV with `pd.read_csv("dataset01.csv")`.
4. Run all cells in order.

## Tools and Libraries

Python, pandas, NumPy, SciPy, scikit-learn, matplotlib, seaborn

## Notes

This project was prepared for academic purposes. It is not a production lending system.
