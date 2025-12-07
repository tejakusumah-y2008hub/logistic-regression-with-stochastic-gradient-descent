import numpy as np


# ---Logistic Regression with Stochastic Gradient Descent ---
class MyLogisticRegression:
    def __init__(
        self,
        n_epochs=1000,
        fit_intercept=True,
        penalty="l2",
        alpha=0.01,
        eta0=0.01,
        t1=10,
    ):
        """
        The 'setup' function.
        """
        self.n_epochs = n_epochs
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.alpha = alpha
        self.eta0 = eta0  # Starting learning rate
        self.t1 = t1  # Cooling parameter for the learning rate

        self.coef_ = None  # To store slopes [m1, m2, ...]
        self.intercept_ = 0.0  # To store intercept [b]
        self._theta = None  # To store the theta (all-in-one list)

    def _learning_schedule(self, t):
        """The "cooling" method for the learning rate"""
        return self.eta0 / (t + self.t1)

    def _sigmoid(self, z):
        """
        The Squeezer helper method.
        Squashes any number to be between 0 and 1.
        """
        # This prevents the code from crashing with overflow warnings
        # z_clipped = np.clip(z, -250, 250)
        # return 1.0 / (1.0 + np.exp(-z_clipped))

        return 1.0 / (1.0 + np.exp(-z))

    def _prepare_X(self, X):
        """
        The "Data Fixer" helper method.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            ones_column = np.ones((X.shape[0], 1))
            return np.hstack([ones_column, X])
        else:
            # Because the intercept is not set to going through the origin,
            # we don't need to add a column of 1s: y = b + mx => y = 0 + mx = mx
            # and we try to find the slopes only.
            return X

    # Create the learn button
    def fit(self, X, y):
        # 1. Prepare X (fix 1D & add '1s' column)
        X_b = self._prepare_X(X)
        n_samples, n_features = X_b.shape

        # 2. Initialize weights
        self._theta = np.zeros(n_features)

        # 3. Stochastic Gradient Descent
        step_counter = 0
        for epoch in range(self.n_epochs):
            # Shuffle indices
            shuffled_indices = np.random.permutation(n_samples)
            X_b_shuffled = X_b[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(n_samples):
                # A. Get ONE sample
                xi = X_b_shuffled[i : i + 1]  # Shape (1, n_features)
                yi = y_shuffled[i : i + 1]  # Shape (1,)

                # B. Calculate prediction for ONE sample
                z = xi @ self._theta
                p = self._sigmoid(z)

                # C. Calculate error for ONE sample
                error = p - yi  # Scalar

                # D. Calculate Gradient for ONE sample
                # Gradient of Cross Entropy wrt weights: xi.T * error
                data_gradient = xi.T @ error

                # E. Regularization (Penalty)
                if self.penalty == "l2":
                    # L2 penalty: alpha * theta
                    penalty_gradient = self.alpha * self._theta
                    if self.fit_intercept:
                        penalty_gradient[0] = 0
                elif self.penalty == "l1":
                    penalty_gradient = self.alpha * np.sign(self._theta)
                    if self.fit_intercept:
                        penalty_gradient[0] = 0
                else:
                    penalty_gradient = 0

                # F. Update weights
                gradient = data_gradient + penalty_gradient

                step_counter += 1
                lr = self._learning_schedule(step_counter)
                self._theta = self._theta - (lr * gradient)

        # 4. Store final results
        if self.fit_intercept:
            self.intercept_ = self._theta[0]
            self.coef_ = self._theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self._theta

        return self

    def predict_proba(self, X):
        """
        The probability guess method.
        """
        # 1. Fix the new data
        X_b = self._prepare_X(X)

        # 2. Calculate the line
        z = X_b @ self._theta

        # 3. Squash it and return the probability
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        The final answer method
        """
        # 1. Get the probabilities
        probabilities = self.predict_proba(X)

        # 2. Apply the threshold
        # (probabilities >= threshold) is a True/False check
        # .astype(int) turns True->1 and False->0
        return (probabilities >= threshold).astype(int)

    def score(self, X, y):
        """
        Calculate the accuracy of the model
        """
        # 1. Get the model's final "0 or 1" guesses
        y_predicted = self.predict(X)

        # 2. Count how many guesses == real answers
        correct_guesses = np.sum(y_predicted == y)

        # 3. Get the total number of data points
        total_guesses = len(y)

        # 4. Return the accuracy
        return correct_guesses / total_guesses
