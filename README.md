# ECM402 Lab 2: Classical Machine Learning for Signal Classification

**Author:** Vivekanand Ojha (Roll No: 723159)

## Track Selection
**Track Chosen:** Track B - Spoken Digit Classification (Free Spoken Digit Dataset)

**Speaker IDs Used:**
The dataset consists of recordings from 6 distinct speakers. To prevent data leakage, the data was grouped and split strictly by speaker identity:
*   **Training Speakers (5):** `george`, `jackson`, `lucas`, `nicolas`, `theo`
*   **Test Speaker (1):** `yweweler` (held out entirely for the final test set evaluation)

## Project Structure
```text
ecm402-lab2-723159/
|-- README.md
|-- requirements.txt
|-- report.pdf
|
|-- src/
|   |-- data.py           # loading + speaker-wise splitting
|   |-- features.py       # handcrafted feature extraction (MFCCs, ZCR, etc.)
|   |-- logistic.py       # from-scratch logistic regression
|   |-- models.py         # SVM / Random Forest training + tuning pipelines
|   |-- evaluate.py       # metrics, significance tests, error analysis
|   `-- main.py           # master script that reproduces all results
|
|-- results/
|   |-- figures/          # Generated confusion matrices and error analysis plots
|   `-- tables/           # Generated CSV metrics and paired t-test results
|
`-- free-spoken-digit-dataset/ # Target directory for the cloned dataset# ecm402-lab2-723159
