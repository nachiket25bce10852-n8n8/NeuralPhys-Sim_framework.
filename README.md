NeuralPhys-Sim 
An Integrated Suite for Multivariable Calculus Visualization and AI-Driven Ballistics

Project OverviewNeuralPhys-Sim is a computational platform developed for the VIT Bhopal BYOP (Bring Your Own Project) capstone. It bridges the gap between theoretical physics and machine learning by providing a "Digital Laboratory."The project features a Numerical Physics Engine to simulate real-world motion, a 3D Visualizer for complex mathematical surfaces, and a Machine Learning Model that learns to predict physical outcomes without solving differential equations.

Key FeaturesPhysics Engine: Uses the Euler Method for numerical integration to simulate projectile motion. Unlike standard formulas, this accounts for Air Resistance (Drag).Math Visualizer: Renders 3D surfaces (e.g., $z = \sin(\sqrt{x^2 + y^2})$) and 2D Contour Maps, essential for understanding gradients and level curves in Multivariable Calculus.AI Predictor: Implements a Polynomial Regression model (Degree 3) that trains on simulated physics data to provide instant impact-range predictions.

Project StructureThe project follows a modular architecture to demonstrate clean version control and software engineering principles:
main.py: The entry point featuring a user-friendly menu system.
physics_engine.py: Contains the logic for numerical integration and data generation.
math_visualizer.py: Handles 3D plotting and calculus visualizations.
ml_predictor.py: Contains the Scikit-Learn implementation of the AI model.
