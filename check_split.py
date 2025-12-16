import os

BASE_DIR = "data"
FOLDERS = ["train", "validation", "test"]
CLASSES = ["ok", "defect"]

total_global = 0

print(f"{'DATASET':<15} | {'OK':<10} | {'DEFECT':<10} | {'TOTAL':<10}")
print("-" * 55)

for folder in FOLDERS:
    count_ok = len(os.listdir(os.path.join(BASE_DIR, folder, "ok")))
    count_defect = len(os.listdir(os.path.join(BASE_DIR, folder, "defect")))
    total = count_ok + count_defect
    total_global += total
    
    print(f"{folder.upper():<15} | {count_ok:<10} | {count_defect:<10} | {total:<10}")

print("-" * 55)
print(f"TOTAL GENERAL: {total_global} imagini")

# Verificare Procente
if total_global > 0:
    print(f"\nProcente estimate:")
    print(f"Train: {len(os.listdir(os.path.join(BASE_DIR, 'train', 'ok'))) * 2 / total_global * 100:.1f}% (Target: 70%)")