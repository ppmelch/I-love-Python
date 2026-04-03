# I-love-Python
Credit Models Homework!
---

This Homework focuses on predicting credit default using machine learning techniques. Two models were implemented and compared: Logistic Regression and XGBoost.

The analysis includes exploratory data visualization, model evaluation, and performance comparison using metrics such as Accuracy and AUC.

---

## Dataset

The dataset contains financial and demographic information of customers, including:

- `creditLimit`
- `gender`
- `edu`
- `age`
- `nDelay`
- `billAmt1` to `billAmt6`
- `default` (target variable)

---

## Objective

To predict whether a customer will default (`default = 1`) based on historical financial behavior and demographic variables.

---

## Models Used

- Logistic Regression
- XGBoost

---

## Evaluation Metrics

- Accuracy
- ROC Curve
- AUC (Area Under the Curve)
- Confusion Matrix

---

## Visualization

A custom `Visualization` class was implemented to generate:

- Scatter plots (feature interaction)
- Histograms
- Boxplots
- ROC curves
- Confusion matrices
- Model comparison plots

---

## Results

| Model               | Accuracy | AUC   |
|--------------------|---------|-------|
| Logistic Regression | ~0.78   | ~0.78 |
| XGBoost             | ~0.79   | **0.81** |

XGBoost achieved better performance in terms of AUC, indicating a stronger ability to distinguish between default and non-default cases.

---

## Key Insights

- Financial variables such as billing amounts (`billAmt1–billAmt6`) and delay history (`nDelay`) are important predictors of default.
- There is no clear separation between classes in the feature space, suggesting that default is driven by complex interactions between variables.
- Both models show limitations in detecting default cases, reflected in the presence of false negatives.

---

## Limitations

- Class overlap makes classification challenging.
- False negatives are significant in a financial risk context.
- No advanced feature engineering was applied.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## Bibliography

  - Python Documentation: https://docs.python.org/3/
  - NumPy Documentation: https://numpy.org/doc/
  - Matplotlib Documentation: https://matplotlib.org/
  - Seaborn Documentation: https://seaborn.pydata.org/
  - Scikit-learn Documentation: https://scikit-learn.org/
  - XGBoost Documentation: https://xgboost.readthedocs.io/

---

## About the Author

This repository was created by **José Armando Melchor Soto**.

---

## Notes

This project demonstrates the application of machine learning techniques to a real-world financial classification problem, highlighting the importance of model evaluation beyond accuracy, especially in imbalanced datasets.

---

## License

This project is licensed under the **MIT License**.  


