import torch

# Model path
model_path = "/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/models/Exp46/best.pth"
output_file = "/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/Arc_model/Exp46_Arc.txt"

# Load model state dictionary
model_state_dict = torch.load(model_path, map_location='cuda:2')

# Store model architecture details in a file
with open(output_file, "w") as file:
    file.write("Model Layers and their dimensions:\n")
    for key, value in model_state_dict.items():
        file.write(f"{key}: {value.shape}\n")

# Confirmation
print(f"Model architecture details have been saved to {output_file}.")
