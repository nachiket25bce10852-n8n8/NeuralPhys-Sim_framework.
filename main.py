import physics_engine as phys  # Shortening names like a real dev
import math_visualizer as math_vis
import ml_predictor as ml
import sys

def start_sim():
    print("==========================================")
    print("   NeuralPhys-Sim V1.0 (BYOP Project)     ")
    print("   Developed for VIT Bhopal - 2nd Sem     ")
    print("==========================================")

    # Initializing the AI brain with degree 3 polynomial
    brain = ml.PhysicsAI()
    
    print("\n[Step 1] Generating physics data for training...")
    try:
        # Generate 600 samples for better accuracy
        angles, vels, ranges = phys.generate_training_data(600)
        
        print("[Step 2] Training the Polynomial Model...")
        acc_score = brain.train_model(angles, vels, ranges)
        print(f"Model Training Complete! Accuracy: {acc_score*100:.2f}%")
        
    except Exception as e:
        print(f"Oops! Something went wrong during startup: {e}")
        sys.exit()

    # The User Interface Loop
    while True:
        print("\n--- OPTIONS MENU ---")
        print("1. Show 3D Calculus Surface (Visual)")
        print("2. Predict Landing Range (AI Engine)")
        print("3. Exit Program")
        
        user_choice = input("\nSelect an option (1, 2, or 3): ").strip()

        if user_choice == '1':
            print("Opening Graph Window... (Please close it to return to menu)")
            math_vis.plot_calculus_surface()
            
        elif user_choice == '2':
            print("\n--- AI Range Predictor ---")
            try:
                # Taking user inputs for prediction
                u_ang = float(input("Enter Launch Angle (10-80 deg): "))
                u_vel = float(input("Enter Launch Velocity (m/s): "))
                
                # Logic check for realistic values
                if u_ang < 0 or u_ang > 90:
                    print("Error: Angle should be between 0 and 90.")
                    continue
                
                prediction = brain.predict_range(u_ang, u_vel)
                print(f"-> Result: The AI thinks it will land at {prediction:.2f} meters.")
                
            except ValueError:
                print("Error: Please enter numbers only!")

        elif user_choice == '3':
            print("Shutting down... Goodbye!")
            break
            
        else:
            print("Invalid choice! Please pick 1, 2, or 3.")

# Running the script
if __name__ == "__main__":
    start_sim()
