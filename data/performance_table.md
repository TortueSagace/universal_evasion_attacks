| Question | Linear | Ridge | Lasso | KNeighbors | SVR | DecisionTree | RandomForest | GradientBoosting | XGB | AdaBoost | ExtraTrees |
|-------------------------------------------------|---|---|----|---|---|-----|-----|-----|------|-----|-----|
| Linear relationship assumed?                    | 2 | 2 |  2 | 0 | 1 |  0  |  0  |  0  |   0  |  0  |  0  |
| Outcome variable continuous?                    | 2 | 2 |  2 | 2 | 2 |  2  |  2  |  2  |   2  |  2  |  2  |
| Model transparency/interpretability important?  | 2 | 1 |  1 | 1 | 0 |  2  |  1  |  1  |   1  |  1  |  1  |
| Homoscedasticity assumed?                       | 2 | 2 |  2 | 1 | 1 |  1  |  1  |  1  |   1  |  1  |  1  |
| Dataset small?                                  | 1 | 1 |  1 | 2 | 2 |  2  |  1  |  1  |   1  |  1  |  1  |
| Dataset large?                                  | 2 | 2 |  2 | 0 | 0 |  1  |  2  |  2  |   2  |  1  |  2  |
| Independent variables not highly correlated?    | 2 | 2 |  2 | 1 | 1 |  1  |  1  |  1  |   1  |  1  |  1  |
| Small influence of outliers on model?           | 0 | 2 |  2 | 0 | 2 |  0  |  1  |  1  |   1  |  0  |  1  |
| Benchmark model needed?                         | 2 | 2 |  2 | 1 | 1 |  1  |  1  |  1  |   1  |  1  |  1  |
| Computationally fast model preferred?           | 2 | 1 |  1 | 0 | 0 |  2  |  0  |  0  |   0  |  1  |  0  |
| Establishing causality a requirement?           | 2 | 2 |  2 | 1 | 1 |  1  |  1  |  1  |   1  |  1  |  1  |
| High degree of multicollinearity in data?       | 0 | 2 |  2 | 1 | 1 |  1  |  1  |  1  |   1  |  1  |  1  |
| Risk of overfitting model?                      | 0 | 2 |  2 | 1 | 2 |  0  |  1  |  1  |   1  |  0  |  1  |
| Large number of predictors?                     | 0 | 2 |  2 | 0 | 2 |  1  |  2  |  2  |   2  |  1  |  2  |
| Managing bias-variance tradeoff important?      | 0 | 2 |  2 | 1 | 2 |  1  |  2  |  2  |   2  |  1  |  2  |
| Preventing ill-conditioning in model important? | 0 | 2 |  2 | 1 | 2 |  1  |  2  |  2  |   2  |  1  |  2  |
| Shrinkage of coefficients required?             | 0 | 2 |  2 | 0 | 2 |  0  |  0  |  2  |   2  |  0  |  0  |
| Regularization of parameter tuning necessary?   | 0 | 2 |  2 | 0 | 2 |  0  |  1  |  2  |   2  |  0  |  1  |
| Simplicity/ease of understanding model valued?  | 2 | 0 |  1 | 1 | 0 |  2  |  0  |  0  |   0  |  0  |  0  |
| Less computationally intensive model preferred? | 2 | 0 |  1 | 0 | 0 |  1  |  0  |  0  |   0  |  1  |  0  |
| More features than observations in data?        | 0 | 1 |  2 | 0 | 2 |  1  |  2  |  2  |   2  |  1  |  2  |
| Feature elimination to reduce complexity?       | 0 | 0 |  2 | 0 | 1 |  0  |  0  |  1  |   1  |  0  |  0  |
| Non-linear relationship in dataset?             | 0 | 0 |  0 | 2 | 2 |  2  |  2  |  2  |   2  |  1  |  2  |
| Local information in dataset important?         | 0 | 0 |  0 | 2 | 1 |  1  |  1  |  1  |   1  |  0  |  1  |
| Data relatively noise-free, few outliers?       | 1 | 1 |  1 | 0 | 2 |  0  |  1  |  1  |   1  |  0  |  1  |
| Kernel-based non-linear flexibility required?   | 0 | 0 |  0 | 0 | 2 |  0  |  0  |  1  |   2  |  0  |  0  |
| Handles high-dimensional data effectively?      | 0 | 0 |  0 | 0 | 2 |  0  |  2  |  2  |   2  |  0  |  2  |
| Margin of tolerance in regression important?    | 0 | 0 |  0 | 0 | 2 |  0  |  0  |  0  |   0  |  0  |  0  |
| Interpretability with a white-box model required?| 2 | 1 |  1 | 1 | 0 |  2  |  1  |  1  |   1  |  1  |  1  |