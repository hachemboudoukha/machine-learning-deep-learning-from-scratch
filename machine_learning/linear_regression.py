import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """
        Entraîne le modèle de régression linéaire.

        Args:
            X (numpy.ndarray): Les caractéristiques d'entrée.
            y (numpy.ndarray): Les valeurs cibles.
        """
        X = np.array(X)
        y = np.array(y)

        # Ajouter une colonne de uns pour le terme d'interception
        X_b = np.c_[np.ones((len(X), 1)), X]

        # Calculer les coefficients en utilisant l'équation normale
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        self.intercept = theta_best[0]
        self.coefficients = theta_best[1:]

    def predict(self, X):
        """
        Effectue des prédictions à l'aide du modèle entraîné.

        Args:
            X (numpy.ndarray): Les caractéristiques d'entrée pour les prédictions.

        Returns:
            numpy.ndarray: Les prédictions.
        """
        X = np.array(X)
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b.dot(np.r_[self.intercept, self.coefficients])

    def __repr__(self):
        return f"LinearRegression(coefficients={self.coefficients}, intercept={self.intercept})"

# Exemple d'utilisation
if __name__ == "__main__":
    # Générer des données d'exemple
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Créer et entraîner le modèle
    model = LinearRegression()
    model.fit(X, y)

    # Effectuer des prédictions
    X_new = np.array([[0], [2]])
    y_predict = model.predict(X_new)

    print("Prédictions:", y_predict)
    print("Modèle:", model)