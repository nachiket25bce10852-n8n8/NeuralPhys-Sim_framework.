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