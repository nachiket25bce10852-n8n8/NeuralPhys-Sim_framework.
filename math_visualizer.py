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