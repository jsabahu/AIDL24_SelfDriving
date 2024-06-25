import torch
from models.model_mask_R_CNN import LaneDetectionModel
from models.LaneNet.LaneNet import LaneNet
from torchvision import transforms
from utils import generate_full_image_rois, calculate_rotation_difference
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define Target Size
#target_size = (480, 854) # models/train_mask_rCNN_33e_70Ks (480, 854).pth
target_size = (180,320) # train_mask_rCNN_30e_50Ks (180, 320).pth

# Prepare Mask R-CNN model
model1 = LaneDetectionModel().to(DEVICE)
model1.load_state_dict(torch.load('models/train_mask_rCNN.pth', map_location=DEVICE))
model1.eval()

# Transform for Mask R-CNN
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(target_size, antialias=True)
])

# Prepare LaneNet model
model2 = LaneNet().to(DEVICE)
model2.load_state_dict(torch.load('models/Lane_Model_ENet.pth', map_location=DEVICE))
model2.eval()

# Transform for LaneNet
transform2 = transforms.Compose([
    #transforms.Resize((target_size[0], target_size[1])),
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Open the video file
video_file = 'data/Example1.mp4'
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
print(f"Video {video_file} is loaded.")

# Get frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video acquired at {fps:.2f} frames per second.")

# Calculate the interval in terms of frames (1 frame per second)
interval = int(fps)
print(f"The video will be processed every {interval} frames to have a 1-second refresh.")

# Initialize video writer
output_video_file = 'data/Example1_output.mp4'
output_video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (2*target_size[1]*3, 2*target_size[0]))

# Create the plot window
fig, axes = plt.subplots(1, 3, figsize=(2*target_size[1]*0.03, 2*target_size[0]*0.01))
#fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
im_original = axes[0].imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
axes[0].set_title("Original Frame")
axes[0].axis('off')
im_predicted_mask_rCNN = axes[1].imshow(np.ones((target_size[0], target_size[1]), dtype=np.uint8), cmap='gray', vmin=0, vmax=1)
axes[1].set_title("Predicted Mask rCNN")
axes[1].axis('off')
im_predicted_LaneNet = axes[2].imshow(np.ones((target_size[0], target_size[1]), dtype=np.uint8), cmap='gray', vmin=0, vmax=1)
axes[2].set_title("Predicted Mask LaneNet")
axes[2].axis('off')
annotation1 = axes[1].text(160, 20, "0", ha='center', va='center', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
annotation2 = axes[2].text(160, 20, "0", ha='center', va='center', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
plt.tight_layout()
plt.show(block=False)  # Show the plot window without blocking

# Loop until the end of the video
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        frame_resized = cv2.resize(frame, target_size)
        image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        #image = Image.open("data\\imagen.png").convert("RGB")   # Image for validation using plot blocked

        # Mask R-CNN processing
        image1 = transform1(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # Store previous image
            if frame_count != 0:
                output1_np_prev = output1_np
            # New image
            output1 = model1(image1, generate_full_image_rois(1, target_size, 0))
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=DEVICE).view(1, 3, 1, 1)
            output1 = (output1 * weights).sum(dim=1, keepdim=True)
            output1 = F.interpolate(output1, size=target_size, mode="bilinear", align_corners=False)
            output1 = (output1 - output1.min()) / (output1.max() - output1.min())
            output1 = (output1 > 0.001).type(torch.int)
            output1_np = output1.detach().cpu().numpy()[0].transpose(1, 2, 0)

        # LaneNet processing
        image2 = transform2(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # Store previous image
            if frame_count != 0:
                output2_np_prev = output2_np
            # New image
            output2 = model2(image2)
            output2 = torch.squeeze(output2["instance_seg_logits"],dim=0).transpose(1, 2).transpose(0, 2)
            weights2 = torch.tensor([0.2989, 0.5870, 0.1140], device=DEVICE).view(1, 1, 3)
            output2 = (output2 * weights2).sum(dim=2, keepdim=True)
            output2 = (output2 - output2.min()) / (output2.max() - output2.min())
            output2 = (output2 > 0.3).type(torch.int)
            output2_np = output2.detach().cpu().numpy()

        # Show angle difference
        if frame_count != 0:
            angle1 = calculate_rotation_difference(output1_np_prev, output1_np)
            angle2 = calculate_rotation_difference(output2_np_prev, output2_np)
        else:
            angle1 = 0
            angle2 = 0

        annotation1.set_text(f"Rotation: {angle1:.2f} deg")
        annotation2.set_text(f"Rotation: {angle2:.2f} deg")

        #print(f"Estimated rotation difference: {angle:.2f} degrees")
        # Update the Matplotlib plot
        #im_original.set_data(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        im_original.set_data(image)
        im_predicted_mask_rCNN.set_data(output1_np)
        im_predicted_LaneNet.set_data(output2)

        # Refresh the plot
        plt.pause(0.01)  # Pause for a short interval to update plot

        # Save plot as frame
        fig.canvas.draw()
        plot_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_frame = plot_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_frame = cv2.cvtColor(plot_frame, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        output_video.write(plot_frame)

        # Debug prints
        #print(output2["instance_seg_logits"].shape)
        #print(output2["binary_seg_pred"].shape)
        #print(output2["binary_seg_logits"].shape)
        #print(output2_np.min(),output2_np.max())
        #print("0:",(100*torch.sum(output2_np == 0).item())/output2_np.numel())
        #print("1:",(100*torch.sum(output2_np == 1).item())/output2_np.numel())
        #print(output1_np.shape)
        #print(output2_np.shape)
        
    frame_count += 1

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
output_video.release()

