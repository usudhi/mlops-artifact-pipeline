# MLOps Artifact Pipeline – Digit Classification using Logistic Regression

This project implements an end-to-end **MLOps pipeline** using **GitHub Actions** to automate:
- Model Training
- Unit Testing with Pytest
- Inference Pipeline with CI/CD

##  Objective
Classify handwritten digits using the `sklearn.datasets.load_digits` dataset and `LogisticRegression`. Automate the workflow using GitHub Actions and organize the code into three phases across separate branches.

---

##  Project Structure

```text
.
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py
├── config/
│   └── config.json
├── tests/
│   └── test_train.py
├── .github/
│   └── workflows/
│       ├── train.yml
│       ├── test.yml
│       └── inference.yml
├── requirements.txt
├── README.md
 ```

---

###  **GitHub Actions Workflow Summary**

```markdown


| Workflow        | Branch       | Purpose                            |
|----------------|--------------|------------------------------------|
| `train.yml`    | classification | Trains model and uploads artifact  |
| `test.yml`     | test         | Runs unit tests via pytest         |
| `inference.yml`| inference    | Full pipeline: test → train → inference |
 ```
##  Branching Structure

### main
#### └── classification (training)
#### └── test (unit testing)
#### └── inference (inference and CI/CD)

## 📊 Results

### ✅ Training Results
After training the Logistic Regression model on the `load_digits` dataset:

- **Accuracy**: 0.9750
- **F1 Score**: 0.9741

Model was saved as `model_train.pkl`.

---

### ✅ Inference Results
When running inference on the same dataset using the trained model:

- **Accuracy**: 0.9750
- **F1 Score**: 0.9741
- **Sample Predictions (first 10)**:  
  `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

---

These results confirm that the model is performing well on digit classification.

## Author

- **Name**: Sudheesh U  
- **GitHub**: [usudhi](https://github.com/usudhi)  
- **Roll No.**: G24AI1057  
- **Email**: g24ai1057@iitj.ac.in
