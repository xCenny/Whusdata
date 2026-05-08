import os
import zipfile
import datetime

def create_backup():
    # Sistem yedekleme dosya adini olustur
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"whusdata_backup_{timestamp}.zip"
    
    # Yedeklemeden haric tutulacak klasor ve dosyalar
    exclusions = ['__pycache__', '.venv', 'venv', 'env']
    
    print(f"Yedekleme islemi basliyor: {backup_filename}")
    
    with zipfile.ZipFile(backup_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            # Haric tutulan klasorleri atla
            dirs[:] = [d for d in dirs if d not in exclusions]
            
            for file in files:
                # Zip dosyalarini ve haric tutulanlari atla
                if file.endswith('.zip') or file in exclusions:
                    continue
                
                file_path = os.path.join(root, file)
                
                # Yedek dosyasinin kendisini yedeklemeyi engelle
                if file_path == f".\\{backup_filename}" or file_path == f"./{backup_filename}":
                    continue
                    
                arcname = os.path.relpath(file_path, ".")
                zipf.write(file_path, arcname)
                print(f"Eklendi: {arcname}")

    print(f"\nYedekleme basariyla tamamlandi: {backup_filename}")
    print("Bu zip dosyasini yeni sunucunuza tasiyarak sistemi oraya kurabilirsiniz.")

if __name__ == "__main__":
    create_backup()
