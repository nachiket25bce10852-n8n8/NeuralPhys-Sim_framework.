import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class PhysicsAI:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=3)
        self.model = LinearRegression()

    def train_model(self, angles, velocities, ranges):
        X = np.column_stack((angles, velocities))
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, ranges)
        return self.model.score(X_poly, ranges)

    def predict_range(self, angle, velocity):
        X_input = self.poly.transform([[angle, velocity]])
        return self.model.predict(X_input)[0]