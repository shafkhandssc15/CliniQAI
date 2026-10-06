import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_records=3000):
    print(f"Generating {num_records} synthetic clinical patient profiles...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define classes and their target sizes
    classes = [
        "Healthy", 
        "Anemia", 
        "Infection", 
        "Diabetes", 
        "Chronic Kidney Disease", 
        "Hypothyroidism", 
        "Hyperthyroidism", 
        "Hypercholesterolemia"
    ]
    
    records_per_class = num_records // len(classes)
    
    data_list = []
    
    for cls in classes:
        # Generate basic demographics
        age = np.random.randint(18, 85, size=records_per_class)
        gender = np.random.choice(["Male", "Female"], size=records_per_class)
        
        # Initialize default standard healthy ranges
        # Hemoglobin (Hb): Normal 12.0 - 17.5 g/dL
        # WBC: Normal 4.5 - 11.0 x10^3/uL
        # Platelets: Normal 150 - 450 x10^3/uL
        # Fasting Blood Sugar (FBS): Normal 70 - 100 mg/dL
        # Creatinine: Normal 0.6 - 1.2 mg/dL
        # TSH: Normal 0.4 - 4.5 mIU/L
        # Cholesterol: Normal 120 - 200 mg/dL
        
        hb = np.random.normal(14.5, 1.2, size=records_per_class)
        wbc = np.random.normal(7.0, 1.5, size=records_per_class)
        plt = np.random.normal(280.0, 50.0, size=records_per_class)
        fbs = np.random.normal(85.0, 8.0, size=records_per_class)
        creatinine = np.random.normal(0.9, 0.15, size=records_per_class)
        tsh = np.random.normal(2.2, 0.8, size=records_per_class)
        chol = np.random.normal(160.0, 15.0, size=records_per_class)
        
        # Adjust features based on disease class to model clinical realities
        if cls == "Anemia":
            # Very low hemoglobin
            hb = np.random.normal(8.5, 1.2, size=records_per_class)
            # Sometimes lower platelets or slightly higher WBC, but keep simple
            plt = np.random.normal(220.0, 60.0, size=records_per_class)
            
        elif cls == "Infection":
            # Elevated White Blood Cells (Leukocytosis)
            wbc = np.random.normal(17.5, 4.0, size=records_per_class)
            hb = np.random.normal(13.0, 1.5, size=records_per_class) # slight drop due to acute stress
            
        elif cls == "Diabetes":
            # High Fasting Blood Sugar
            fbs = np.random.normal(185.0, 45.0, size=records_per_class)
            # Clip minimum FBS to pre-diabetic level for this class
            fbs = np.maximum(fbs, 110.0)
            
        elif cls == "Chronic Kidney Disease":
            # Elevated Serum Creatinine
            creatinine = np.random.normal(3.8, 1.6, size=records_per_class)
            # Clip minimum creatinine to elevated level
            creatinine = np.maximum(creatinine, 1.6)
            # Anemia is very common in CKD due to lower erythropoietin production
            hb = np.random.normal(10.5, 1.1, size=records_per_class)
            
        elif cls == "Hypothyroidism":
            # High TSH (underactive thyroid)
            tsh = np.random.normal(16.5, 5.0, size=records_per_class)
            tsh = np.maximum(tsh, 5.0)
            # Often associated with slightly elevated cholesterol
            chol = np.random.normal(220.0, 25.0, size=records_per_class)
            
        elif cls == "Hyperthyroidism":
            # Low TSH (overactive thyroid)
            tsh = np.random.normal(0.08, 0.04, size=records_per_class)
            tsh = np.maximum(tsh, 0.01) # TSH cannot be <= 0
            
        elif cls == "Hypercholesterolemia":
            # Elevated Total Cholesterol
            chol = np.random.normal(275.0, 35.0, size=records_per_class)
            chol = np.maximum(chol, 210.0)
            
        # Compile records
        for i in range(records_per_class):
            data_list.append({
                "Age": int(age[i]),
                "Gender": gender[i],
                "Hemoglobin": round(float(hb[i]), 1),
                "WBC": round(float(wbc[i]), 1),
                "Platelets": int(plt[i]),
                "Fasting_Blood_Sugar": int(fbs[i]),
                "Creatinine": round(float(creatinine[i]), 2),
                "TSH": round(float(tsh[i]), 2),
                "Cholesterol": int(chol[i]),
                "Diagnosis": cls
            })
            
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Make sure target folder exists
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "synthetic_clinical_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} samples and saved to {csv_path}")
    print("\nClass distribution:")
    print(df["Diagnosis"].value_counts())
    
if __name__ == "__main__":
    generate_synthetic_data()
