NeuralPhys-Sim 

Physics-AI Hybrid Simulator for VIT Bhopal (BYOP)

 OverviewNeuralPhys-Sim is a computational tool that integrates Multivariable Calculus, Numerical Physics, and Machine Learning. It solves the "Abstraction Gap" by providing interactive 3D math surfaces and an AI engine that learns to predict complex ballistic trajectories.
 
Key FeaturesPhysics Engine: Simulates motion using the Euler Method, accounting for non-linear Air Resistance.Math Visualizer: Renders 3D functions  with 2D Contour Maps.AI Predictor: Uses Polynomial Regression (Degree 3) to "learn" physics and predict range instantly.

Project Structure
main.py: Entry point & Menu system.

physics_engine.py: Numerical integration & data generation.

math_visualizer.py: 3D plotting & Calculus logic.

ml_predictor.py: Scikit-Learn AI model implementation.

Quick Start
Install Dependencies:
Bash
python -m pip install numpy matplotlib scikit-learn

Run Application:
Bash
python main.py
