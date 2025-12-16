import os
import csv

# Configurare
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUTPUT_CSV = os.path.join(BASE_DIR, "data_log.csv")

def generate_csv_log():
    print("Generare registru CSV pentru date...")
    
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Scriem antetul (Header)
        writer.writerow(['Timestamp', 'Image_Path', 'Label', 'Status'])
        
        # Iteram prin foldere
        for label in ["ok", "defect"]:
            folder_path = os.path.join(RAW_DIR, label)
            if not os.path.exists(folder_path):
                continue
                
            for img_name in os.listdir(folder_path):
                if img_name.endswith(".png"):
                    # Simulam un timestamp sau folosim indexul
                    # Path-ul relativ pentru portabilitate
                    rel_path = os.path.join("data", "raw", label, img_name)
                    
                    # Cod numeric: OK=0, Defect=1
                    status_code = 0 if label == "ok" else 1
                    
                    writer.writerow(["2023-12-11 10:00:00", rel_path, label, status_code])
                    
    print(f"CSV generat cu succes: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_csv_log()