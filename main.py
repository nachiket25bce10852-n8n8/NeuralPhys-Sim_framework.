import physics_engine
import math_visualizer
import ml_predictor

def main():
    print("--- Welcome to NeuralPhys-Sim (BYOP Project) ---")
    ai = ml_predictor.PhysicsAI()
    
    # 1. Pre-train AI using Physics Engine
    a, v, r = physics_engine.generate_training_data(500)
    acc = ai.train_model(a, v, r)
    
    while True:
        print("\n1. Visualise 3D Math Surface\n2. AI Range Prediction\n3. Exit")
        choice = input("Select: ")
        
        if choice == '1':
            math_visualizer.plot_calculus_surface()
        elif choice == '2':
            ang = float(input("Angle (10-80): "))
            vel = float(input("Velocity (m/s): "))
            pred = ai.predict_range(ang, vel)
            print(f"AI Predicted Range: {pred:.2f} meters (Accuracy: {acc*100:.1f}%)")
        elif choice == '3':
            break

if __name__ == "__main__":
    main()

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


import numpy as np
import matplotlib.pyplot as plt

def plot_calculus_surface():
    """Visualizes z = f(x, y) with contours."""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Example: A 'Ripple' function (Multivariable Calculus)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Adding a contour plot (Level Curves)
    ax.contour(X, Y, Z, zdir='z', offset=-1.2, cmap='viridis')
    
    ax.set_title("3D Surface & Level Curves (Calculus)")
    plt.show()


import numpy as np

def simulate_projectile(v0, angle, drag_coeff=0.1, mass=1.0):
    """Calculates the range of a projectile considering air resistance."""
    g = 9.81
    dt = 0.1
    theta = np.radians(angle)
    
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x, y = 0.0, 0.0
    
    # Numerical integration (Euler Method)
    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        # Drag force approximation
        drag = 0.5 * drag_coeff * v**2
        ax = -(drag * (vx/v)) / mass
        ay = -g - (drag * (vy/v)) / mass
        
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        
        if x > 2000: break # Safety break
    return x

def generate_training_data(n_samples=100):
    """Generates a dataset for the AI to learn from."""
    angles = np.random.uniform(10, 80, n_samples)
    velocities = np.random.uniform(10, 100, n_samples)
    ranges = [simulate_projectile(v, a) for v, a in zip(velocities, angles)]
    return angles, velocities, ranges