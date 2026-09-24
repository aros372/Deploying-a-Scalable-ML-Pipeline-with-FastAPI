# Model Card

## Model Details
This model uses the Random Forest Classifier from scikit-learn version 1.9.1 with default hyperparameters. 

## Intended Use
The intended use is to classify the data by binary income (above or below $50K/yr).

## Dataset
Census Income - Donated on 4/30/1996
Extraction was done by Barry Becker from the 1994 Census database.  A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
https://archive.ics.uci.edu/dataset/20/census+income

## Training Data
The training data was split from the full dataset and preprocessed before being used to train the model.

## Evaluation Data
The evaluation data was split from the full dataset before any adjustments were made and has the same features and label schema.

## Metrics
The model was tested on precision, recall, and F1-score metrics. 
The metric outcomes showed that most of the positive predictions are correct (74% precision) though it misses many positive cases (63% recall) but it is relatively balanced (0.68 F1-score).
Exact outcomes -> Precision: 0.7425 | Recall: 0.6334 | F1: 0.6836

## Ethical Considerations
The dataset is 30 years old and there's little documentation on collection conditions and considerations so there may be hidden biases and/or incorrect data especially regarding racial and geographic biases.

## Caveats and Recommendations
Performance can be better refined with k-fold cross-validation and tuned hyperparameters. Additionally, the results would be more practically useful with newer data, because as it stands, it would not be recommended to use for high-stakes, real-world decisions.