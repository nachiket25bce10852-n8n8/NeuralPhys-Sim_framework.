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