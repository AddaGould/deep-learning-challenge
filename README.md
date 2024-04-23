# deep-learning-challenge

## Analysis Report
The purpose of this analysis is to develop a binary classifier using a sequential model to predict whether organizations funded by Alphabet Soup will be successful based on various features provided in the dataset. The dataset contains information about more than 34,000 organizations, including metadata such as application type, affiliation, classification, and income amount.

## Results
Data Preprocessing
* Target Variable(s): The target variable for the model is "IS_SUCCESSFUL," which indicates whether the funding provided by Alphabet Soup was used effectively (1 for successful, 0 for unsuccessful).

* Feature Variable(s): The feature variables for the model include metadata such as application type, affiliation, classification, use case, organization type, income amount, and special considerations.

* Removed Variable(s): The EIN and NAME columns were dropped from the dataset as they are identification columns and not relevant for modeling.

![Screenshot 2024-04-23 133321](https://github.com/AddaGould/deep-learning-challenge/assets/148924005/32d528c8-aae3-461f-8bda-3a8cf8c37f1e)

## Compiling, Training, and Evaluating the Model
* Neural Network Model Design: The neural network model consists of an input layer, one or more hidden layers, and an output layer. The number of neurons, layers, and activation functions were adjusted to optimize the model's performance.

* Model Architecture: The model architecture varied across attempts, including changes in the number of neurons, layers, and activation functions. Attempts included increasing the number of neurons in the hidden layers, adding dropout layers for regularization, and exploring different activation functions such as ReLU, Leaky ReLU, ELU, and Tanh.

```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
nn_model = tf.keras.models.Sequential()

# First hidden layer with Leaky ReLU activation
nn_model.add(tf.keras.layers.Dense(units=64, activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_dim=108, kernel_initializer="VarianceScaling"))

# Second hidden layer with ELU activation
nn_model.add(tf.keras.layers.Dense(units=32, activation=tf.keras.layers.ELU(), kernel_initializer="VarianceScaling"))

# Third hidden layer with Tanh activation
nn_model.add(tf.keras.layers.Dense(units=16, activation="tanh", kernel_initializer="VarianceScaling"))

# Output layer with Sigmoid activation for binary classification
nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_initializer="VarianceScaling"))
```

* Achievement of Target Model Performance: Despite several attempts to optimize the model, the performance metrics remained below the target. The model consistently achieved a loss of approximately 0.55 and an accuracy of around 0.73.

![Screenshot 2024-04-23 133628](https://github.com/AddaGould/deep-learning-challenge/assets/148924005/aa720b27-532d-4958-88e8-c561c720fd78)

* Steps Taken to Improve Model Performance: Attempts to improve model performance included adjusting the number of neurons in the hidden layers, adding dropout layers for regularization, and experimenting with different activation functions. However, these modifications did not result in a significant improvement in model performance.

## Summary
The deep learning model developed for this classification problem did not achieve the target predictive accuracy of higher than 75%. Despite various optimization attempts, including adjustments to the model architecture and activation functions, the model's performance remained suboptimal.

A recommendation for addressing this classification problem would be to explore alternative machine learning algorithms and techniques beyond deep learning. Gradient boosting algorithms such as XGBoost or LightGBM could be effective in capturing complex patterns in the data and achieving higher accuracy. Additionally, feature engineering and selection techniques could help improve model performance by identifying the most relevant features for prediction.

Exploring ensemble methods, such as stacking or blending multiple models, could also enhance predictive accuracy by leveraging the strengths of different algorithms. By combining the predictions of multiple models, ensemble methods often outperform individual models and provide more robust predictions.

In conclusion, while sequential learning models offer powerful capabilities for solving complex classification problems, they may not always be the most effective approach. By exploring alternative algorithms and techniques, such as gradient boosting and ensemble methods, we can potentially achieve better performance and provide more accurate predictions for Alphabet Soup's funding selection process.
