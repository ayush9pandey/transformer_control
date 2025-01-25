import os
import pickle

pickle_dir = "dataset_multipendulum_gaussian/saved_pickles"

file_number = 205000 

pickle_file = f"multipendulum_{file_number}.pkl"

file_path = os.path.join(pickle_dir, pickle_file)

if not os.path.exists(file_path):
    print(f"File '{pickle_file}' does not exist in '{pickle_dir}'.")
else:
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        print(f"Contents of {pickle_file}:")
        print(data[1])
    except Exception as e:
        print(f"An error occurred while reading {pickle_file}: {e}")
