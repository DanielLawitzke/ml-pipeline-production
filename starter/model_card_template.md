# Model Card

## Model Details

This model predicts whether an individual earns more than $50K per year based on census data. It uses a Random Forest Classifier implemented with scikit-learn version 1.7.2. The model was trained as part of a machine learning deployment pipeline project.

Training date: November 2025
Model version: 1.0

## Intended Use

The model is intended for educational purposes to demonstrate deployment of a machine learning pipeline. It predicts income level based on demographic and employment features from census data.

The model should not be used for actual decision-making regarding individuals' economic status or creditworthiness due to potential bias in historical data.

## Training Data

The model was trained on the Census Income dataset from the UCI Machine Learning Repository. The dataset contains approximately 32,000 records with 14 features including age, education, occupation, and hours worked per week.

The data was cleaned to remove leading and trailing whitespaces. An 80/20 train-test split was used with random_state=42 for reproducibility.

Categorical features were one-hot encoded and the target variable (salary >50K or <=50K) was binary encoded.

## Evaluation Data

20% of the census data was held out for testing, resulting in approximately 6,500 test samples. The same preprocessing steps applied to training data were applied to the test set.

## Metrics

The model achieved the following performance on the test set:

- Precision: 0.7962
- Recall: 0.5372
- F1 Score: 0.6416

The model shows high precision but lower recall, meaning it is conservative in predicting high earners. It correctly identifies about 80% of predicted high earners but misses about 46% of actual high earners.

Performance varies across demographic slices. For example, the model performs differently across education levels and shows different recall rates for different gender groups.

## Ethical Considerations

The census data reflects historical biases present in society. The model shows different performance metrics across demographic groups, particularly:

- Lower recall for female individuals compared to male individuals
- Varying performance across different education levels and occupations
- Potential reinforcement of existing wage gaps

Users should be aware that predictions may perpetuate historical inequalities present in the training data.

## Caveats and Recommendations

The model has several limitations:

- It was trained on data from 1994 and may not reflect current economic conditions
- The conservative nature of the model means it misses many actual high earners
- Performance varies significantly across demographic groups
- The model should not be used for any real-world decision making about individuals

For production use, the model would need:

- More recent training data
- Fairness constraints to reduce bias across groups
- Regular retraining and monitoring
- More sophisticated feature engineering

The model serves its purpose as a demonstration of ML deployment practices but is not suitable for real-world applications without significant improvements.