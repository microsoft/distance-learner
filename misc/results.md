# Results of comparing distance learner to standard classifier

## k=2, n=500

- 200k samples generated in total. 100k on each manifold. 50k exactly on the manifold for each manifold.
- Standard classifier trained only on exactly on-manifold samples.
- Radii and other characteristics of the spheres taken from the paper.

### Greedy Attack

```
Std. Clf. on actual test set
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     50000
           1       1.00      1.00      1.00     50000

    accuracy                           1.00    100000
   macro avg       1.00      1.00      1.00    100000
weighted avg       1.00      1.00      1.00    100000

Std. Clf. on perturbed test set
              precision    recall  f1-score   support

           0       0.89      0.82      0.85     50000
           1       0.83      0.90      0.86     50000

    accuracy                           0.86    100000
   macro avg       0.86      0.86      0.86    100000
weighted avg       0.86      0.86      0.86    100000

Dist. Learner on actual test set
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     50000
           1       1.00      1.00      1.00     50000

    accuracy                           1.00    100000
   macro avg       1.00      1.00      1.00    100000
weighted avg       1.00      1.00      1.00    100000

Dist. Learner on perturbed test set
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     50000
           1       1.00      1.00      1.00     50000

    accuracy                           1.00    100000
   macro avg       1.00      1.00      1.00    100000
weighted avg       1.00      1.00      1.00    100000
```

### Non-greedy attack

```
Std. Clf. on actual test set
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     50000
           1       1.00      1.00      1.00     50000

    accuracy                           1.00    100000
   macro avg       1.00      1.00      1.00    100000
weighted avg       1.00      1.00      1.00    100000

Std. Clf. on perturbed test set
              precision    recall  f1-score   support

           0       0.00      0.00      0.00   50000.0
           1       0.00      0.00      0.00   50000.0

    accuracy                           0.00  100000.0
   macro avg       0.00      0.00      0.00  100000.0
weighted avg       0.00      0.00      0.00  100000.0

Dist. Learner on actual test set
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     50000
           1       1.00      1.00      1.00     50000

    accuracy                           1.00    100000
   macro avg       1.00      1.00      1.00    100000
weighted avg       1.00      1.00      1.00    100000

Dist. Learner on perturbed test set
              precision    recall  f1-score   support

           0       0.50      0.47      0.48     50000
           1       0.50      0.53      0.51     50000

    accuracy                           0.50    100000
   macro avg       0.50      0.50      0.50    100000
weighted avg       0.50      0.50      0.50    100000
```


### k=2, n=500 concentric spheres trained on 2 million sample dataset

- On-manifold attacks used

```
Std. Clf. on actual test set
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     50000
           1       1.00      1.00      1.00     50000

    accuracy                           1.00    100000
   macro avg       1.00      1.00      1.00    100000
weighted avg       1.00      1.00      1.00    100000

Std. Clf. on perturbed test set by CLF attack
              precision    recall  f1-score   support

           0       0.00      0.00      0.00   50000.0
           1       0.00      0.00      0.00   50000.0

    accuracy                           0.00  100000.0
   macro avg       0.00      0.00      0.00  100000.0
weighted avg       0.00      0.00      0.00  100000.0

Std. Clf. on perturbed test set by DL attack
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     50000
           1       1.00      1.00      1.00     50000

    accuracy                           1.00    100000
   macro avg       1.00      1.00      1.00    100000
weighted avg       1.00      1.00      1.00    100000

Dist. Learner on actual test set
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     50000
           1       1.00      1.00      1.00     50000

    accuracy                           1.00    100000
   macro avg       1.00      1.00      1.00    100000
weighted avg       1.00      1.00      1.00    100000

Dist. Learner on perturbed test set by CLF attack
              precision    recall  f1-score   support

           0       1.00      0.10      0.18     50000
           1       0.00      0.00      0.00     50000
           2       0.00      0.00      0.00         0

    accuracy                           0.05    100000
   macro avg       0.33      0.03      0.06    100000
weighted avg       0.50      0.05      0.09    100000

Dist. Learner on perturbed test set by DL attack
              precision    recall  f1-score   support

           0       0.00      0.00      0.00   50000.0
           1       0.00      0.00      0.00   50000.0
           2       0.00      0.00      0.00       0.0

    accuracy                           0.00  100000.0
   macro avg       0.00      0.00      0.00  100000.0
weighted avg       0.00      0.00      0.00  100000.0

```