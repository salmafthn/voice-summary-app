from datasets import Dataset, DatasetDict, load_from_disk
import json
import os

def load_json_files_from_folder(folder_path, existing_ids=set()):
    """
    Fungsi untuk memuat file JSON dari folder dan menghindari duplikasi berdasarkan 'id'.
    folder_path: Lokasi folder tempat file JSON berada.
    existing_ids: Set yang berisi ID yang sudah ada untuk menghindari duplikasi.
    """
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                entry = json.load(f)
                if entry["id"] not in existing_ids:  # Memeriksa jika ID sudah ada
                    data.append(entry)
                    existing_ids.add(entry["id"])  # Tambahkan ID ke set
    return data, existing_ids

# Path to the saved dataset
dataset_path = "D:/voice-summary-app/data/dataset"

# Initialize a set to keep track of existing IDs
existing_ids = set()

# Check if the dataset already exists on disk
if os.path.exists(dataset_path):
    print("Dataset already exists, loading the dataset...")
    dataset = load_from_disk(dataset_path)
    
    # Extract existing IDs from the current dataset
    for entry in dataset["train"]:
        existing_ids.add(entry["id"])
else:
    print("Dataset does not exist, creating a new dataset...")

# Load new data and avoid duplicates based on "id"
train_data, existing_ids = load_json_files_from_folder("D:/voice-summary-app/data/raw/train", existing_ids)
test_data, existing_ids = load_json_files_from_folder("D:/voice-summary-app/data/raw/test", existing_ids)

# Convert the data into Hugging Face dataset format
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# Create a dataset dictionary for training and testing
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Save the dataset for later use
dataset.save_to_disk(dataset_path)
print("Dataset has been saved to disk.")
