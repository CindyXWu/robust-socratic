"""Print out the final accuracies of the trained teachers."""
import torch
import os

def load_model_and_get_accs(model_path: str) -> float:
    checkpoint = torch.load(model_path)
    return checkpoint['final_acc']


def print_final_accs(folder_path: str) -> None:
    for filename in os.listdir(folder_path):
        if filename.startswith("RN18AP"):
            final_acc = load_model_and_get_accs(f"{folder_path}/{filename}")
            print(f"Model {filename} has final accuracy of: {final_acc}")
            

if __name__ == "__main__":
    folder_path = "Lent/trained_teachers_old"
    print_final_accs(folder_path)