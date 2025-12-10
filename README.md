# Interpreting-Random-Forests-on-a-Heart-Disease-Dataset

This project is a tutorial-style analysis that shows how to interpret a Random Forest classifier trained to predict heart disease using the **Heart Failure Prediction** dataset from Kaggle. The focus is not only on achieving good predictive performance, but on **explaining** what the model has learned through:

- Global feature importance
- 1D and 2D partial dependence plots (PDPs)
- Discussion of limitations, ethics, and accessibility

The material is designed so that a beginner in machine learning can follow the explanations, while still meeting the requirements of an advanced ML/neural networks course.

---

## 1. Project structure

- `Interpreting Random Forests on Heart Disease Data.ipynb` – the main Jupyter notebook. Running all cells reproduces all plots used in the tutorial.
- `data/heart.csv`- dataset.
- `Interpreting Random Forests on a Heart Disease Dataset.pdf` – written tutorial.
- `LICENSE` – MIT.

---

## 2. Dataset

**Name:** Heart Failure Prediction Dataset  
**Source:** Kaggle – https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

The dataset contains 918 patients with 11 input features (age, blood pressure, cholesterol, ECG and exercise information) and a binary target `HeartDisease` indicating presence (1) or absence (0) of heart disease.

To obtain the data:

1. Create/log in to a Kaggle account.
2. Visit the dataset page:  
   https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
3. Download `heart.csv`.
4. Place `heart.csv` in the `data/` folder of this repository.

Please respect the dataset licence and Kaggle’s terms of use.

---

## 3. How to run the notebook

### 3.1. Set up environment

You can use Python 3.9+ and create a virtual environment (recommended):


python -m venv venv
source venv/bin/activate # on Windows: venv\Scripts\activate
pip install the below dependencies:
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter

### 3.2. Run the notebook


jupyter notebook notebooks/ Interpreting Random Forests on Heart Disease Data.ipynb

Then:

1. Ensure `data/heart.csv` exists (or adjust the `data_path` variable at the top of the notebook).
2. Run all cells in order (`Kernel → Restart & Run All`).
3. The notebook will:
   - Load and summarise the dataset.
   - Visualise feature distributions by heart‑disease status.
   - Split data into train/test sets.
   - Build a preprocessing + `RandomForestClassifier` pipeline.
   - Evaluate accuracy, confusion matrix, and classification report.
   - Compute and plot feature importances.
   - Generate 1D and 2D partial dependence plots.
   - Discuss limitations and ethical aspects in markdown cells.

All figures used in the report can be exported from the notebook with `plt.savefig(...)` into the `figures/` folder.

---

## 4. Tutorial / report

The detailed written tutorial is in:

- `Interpreting Random Forests on a Heart Disease Dataset.pdf`  

It explains, in a beginner‑friendly way:

- What decision trees and Random Forests are (with a conceptual diagram).
- How the confusion matrix, TP/TN/FP/FN, accuracy, precision, and recall are defined.
- Which features are most important for this model?
- How PDPs show the effect of `Age`, `MaxHR`, `Oldpeak`, and the interaction between `Age` and `MaxHR`.
- Limitations of feature importance and PDPs.
- Ethical and accessibility considerations when deploying ML in healthcare.

---

## 5. Accessibility

This project includes several features aimed at improving accessibility:

- All plots use a **colour‑blind‑friendly** palette with good contrast.
- Large font sizes in figures and notebook cells.
- Each figure in the report has **alt text** or a detailed caption describing the important information.
- The report is structured with clear headings so that screen‑reader users can navigate it easily.
- If a video tutorial is provided, it should include **captions** and be accompanied by a transcript.

---

## 6. Re-use and licence

The code in this repository is released under the **MIT License** (see `LICENSE`). You are free to reuse and adapt the code with appropriate attribution.

The **dataset** is provided by the Kaggle author and is subject to Kaggle’s licence and terms of use; please check the dataset page before redistributing the data.

---

## 7. References

- Kaggle. *Heart Failure Prediction Dataset (heart.csv).* 2021.  
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, 12, 2011.  
- Molnar, C. *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*, 2nd ed., 2022.  
- Rasheed, K., Qureshi, R., Qamar, A. M., et al. “Explainable, trustworthy, and ethical machine learning for healthcare,” *Computers in Biology and Medicine*, 145, 105403, 2022.  
- Chen, I. Y., Pierson, E., Rose, S., Joshi, S., Ferryman, K., & Ghassemi, M. “Ethical machine learning in healthcare,” *Annual Review of Biomedical Data Science*, 4, 123–144, 2021.  
- Hoche, M., et al. “What makes clinical machine learning fair? A practical ethics framework,” *PLOS Digital Health*, 4(3), e0000728, 2025.

