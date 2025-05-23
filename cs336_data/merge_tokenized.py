import os
import numpy as np
from tqdm import tqdm 

# Specify the directory path containing the .bin files
directory_path = '/data/c-salzhu/tokenizedCC_strict_0522_5/'

# Specify the output file name
output_file = '/data/c-salzhu/CC_tokenized_full_0522_5.bin'

# List to store all the arrays
all_arrays = []

# Iterate through all files in the directory
for filename in tqdm(os.listdir(directory_path)):
    if filename.endswith('.bin'):
        file_path = os.path.join(directory_path, filename)
        
        # Load the NumPy array from the .bin file
        array = np.fromfile(file_path, dtype=np.uint16)  # Adjust dtype if necessary
        
        # Append the array to the list
        all_arrays = np.concatenate([all_arrays, array])

# Combine all arrays into a single large array
# combined_array = np.concatenate(all_arrays)

# Save the combined array to a new .bin file
output_path = os.path.join(directory_path, output_file)
combined_array = all_arrays
combined_array.tofile(output_path)

print(f"Combined array saved to: {output_path}")
print(f'Num tokens: {len(combined_array)}')
print(f"Shape of all array: {all_arrays.shape}, {combined_array.shape}")
