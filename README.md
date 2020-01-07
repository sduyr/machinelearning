# machinelearning

Prediction Ratings from text 

This project is an exercise in using supervised machine learning algorithms to predict
rating scores from restaurant reviews. In practice, machine learning models do not
provide insight into why or how they touched base at a result. In our trials, we
experiment with variables associated with each model in order to find the best classifiers
for our dataset types. In this way, we are given some insight and a better understanding
of how these variables allow us to make better predictions




We were provided with files, training.txt and test.txt, each containing review text and
ratings scores associated with each review. The reviews were taken from outlier detection
datasets, the source of which is cited below.
All proceedings were carried out using either built in Python packages or sklearn, a
library of Python tools created specifically for machine learning undertakings. In order to
pre-process the datasets, the features (reviews and ratings) were extracted from the review
text files. The review features were vectorized using TD-IDF vectorization, which
implements the bag-of-words model we discussed in class.
The next requirement was highly experimental. We were tasked to execute 5-fold cross
validation on our training data and record our observations, while varying several
configuration options (as we saw fit) using each of the following learning models:
1. Neural networks
a. Hidden layers (1, 2, 3)
b. Units per hidden layer (5, 10, 15, 20, 100)
2. Naïve Bayes (no suggested variations)
3. Logistic Regression
a. Penalty values (L1/L2 regularization)
b. Regularization strength (0.001, 0.01, 0.1, 1, 10, 100)
4. AdaBoosting (varied number of estimators 25, 50, 75)
5. Support Vector Machines
a. Kernel types (linear, polynomial, rbf, sigmoid)
b. Cost factor (1, 10, 100, 1000)


Next, we repeated the experiment and recorded our observations after filtering the
datasets with “sentiment” words, which were also provided to us in text files and loaded
into our Python environment.
With the original training data (no filtering) and the optimal configurations obtained, we
trained our classifiers for each learning model. With each classifier, we evaluated the test
review data to predict ratings. We then used the predictions to reporte precision, recall,
and f1 scores against each actual rating value. This information allowed us to observe
which ratings were hardest to predict and reach some conclusions regarding our
experience.
