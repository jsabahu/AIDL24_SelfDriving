import torch
from utils import show_sample, generate_full_image_rois
from models.model_mask_R_CNN import LaneDetectionModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
hparams = {"target_size": (180, 320)}

# Choose a image & mask
image_path = "data\\bdd100k\\images\\100k\\test\\6558820b-6e0594fa.jpg"
mask_path = "data\\bdd100k\\labels\\lane\\masks\\test\\6558820b-6e0594fa.png"

# Load the model
model = LaneDetectionModel()
model.load_state_dict(torch.load('models/train_mask_rCNN.pth'))
model.eval()

# Show the output mask
show_sample(model,image_path,mask_path,hparams["target_size"],DEVICE)