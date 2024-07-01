import torch
from models.model_mask_R_CNN import LaneDetectionModel
from models.LaneNet.LaneNet import LaneNet
import models.model_Faster_R_CNN as FCnn  
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from utils import generate_full_image_rois, calculate_rotation_difference
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#GraphMode = 0 # Just Video with WheelDrive
#GraphMode = 1 # Compare models
#GraphMode = 2 # All
#GraphMode = 3 # Just Video with WheelDrive (Image vs Model) 
#GraphMode = 4 # Compare models overwriting lines
GraphMode = 5 # Just Video with WheelDrive and Car Detection

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
    #transforms.Resize((360, 640)),
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Prepare Faster R-CNN
config = FCnn.load_config() # Load configuration

model3 = FCnn.create_model(config["hyperparameters"]["num_classes"])
model3.load_state_dict(torch.load('models/Faster_R_CNN.pth', map_location=DEVICE))
model3.eval()

# Transform for Faster R-CNN
transform3 = FCnn.create_transforms(config)

# Open the video file
#video_file = 'data/Example1.mp4'
video_file = 'data/video1.hevc'
#video_file = 'data/sample2.mp4'
wheel_file = 'data/drivewheel.png'
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
output_video_file = 'data/Sample_output.mp4'
if GraphMode == 0 or GraphMode == 3 or GraphMode == 5:
    output_video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (2*target_size[1], 2*target_size[0]))
if GraphMode == 1 or GraphMode == 2:
    output_video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (2*target_size[1]*3, 2*target_size[0]))
if GraphMode == 4:
    output_video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (2*target_size[1]*2, 2*target_size[0]))


# Initialize angle
angle = 0
angle_img = 0

# Create the plot window
if GraphMode == 0:
    fig, axes = plt.subplots(1, 1, figsize=(2*target_size[1]*0.01, 2*target_size[0]*0.01))
    #fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    im_original = axes.imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
    axes.set_title("Original Frame")
    axes.axis('off')

if GraphMode == 1:
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

if GraphMode == 2:
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

if GraphMode == 3:
    fig, axes = plt.subplots(1, 1, figsize=(2*target_size[1]*0.01, 2*target_size[0]*0.01))
    #fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    im_original = axes.imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
    axes.set_title("Original Frame")
    axes.axis('off')
    annotation0 = axes.text(160, 20, "0", ha='center', va='center', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

if GraphMode == 4:
    cmap_colour = LinearSegmentedColormap.from_list('colour', [(0, 'red'), (1, 'white')])
    fig, axes = plt.subplots(1, 2, figsize=(2*target_size[1]*0.02, 2*target_size[0]*0.01))
    #fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    im_original1 = axes[0].imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
    #im_mask1 = axes[0].imshow(np.zeros((target_size[0], target_size[1]), dtype=np.uint8), cmap='gray', vmin=0, vmax=1, alpha=0.1)
    im_mask1 = axes[0].imshow(np.zeros((target_size[0], target_size[1]), dtype=np.uint8), cmap=cmap_colour, vmin=0, vmax=1, alpha=0.2)
    axes[0].set_title("Mask rCNN")
    axes[0].axis('off')
    im_original2 = axes[1].imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
    im_mask2 = axes[1].imshow(np.zeros((target_size[0], target_size[1]), dtype=np.uint8), cmap=cmap_colour, vmin=0, vmax=1, alpha=0.2)
    axes[1].set_title("LaneNet")
    axes[1].axis('off')

if GraphMode == 5:
    fig, axes = plt.subplots(1, 1, figsize=(2*target_size[1]*0.01, 2*target_size[0]*0.01))
    #fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    im_original = axes.imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
    axes.set_title("Original Frame")
    axes.axis('off')

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
        # Prepare Wheel Image
        wheel_image = Image.open(wheel_file).convert("RGBA")
        transform = transforms.ToTensor()
        wheel_image = transform(wheel_image)
        wheel_image_resized = transforms.Resize((100, 100))(wheel_image)
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
            output2 = (output2 > 0.5).type(torch.int)
            output2_np = output2.detach().cpu().numpy()

        # Show angle difference       
        if frame_count != 0:
            angle0 = calculate_rotation_difference(image_prev, image, type=1)
            angle1 = calculate_rotation_difference(output1_np_prev, output1_np)
            angle2 = calculate_rotation_difference(output2_np_prev, output2_np)
        else:
            angle0 = 0
            angle1 = 0
            angle2 = 0

        image_prev = image
        
        if GraphMode == 2:
            annotation1.set_text(f"Rotation: {angle1:.2f} deg")
            annotation2.set_text(f"Rotation: {angle2:.2f} deg")

        if GraphMode == 3:
            annotation0.set_text(f"Image: {angle0:.2f} deg vs Model: {angle1:.2f}")
            
        if GraphMode == 0 or GraphMode == 2 or GraphMode == 5:
            # Show WheelDrive
            x_offset = 40
            y_offset = 220

            angle = angle + angle1*2
            rotate_transform = transforms.functional.rotate
            wheel_image_rotated = rotate_transform(wheel_image_resized, angle=angle)

            wheel_image_rgb = wheel_image_rotated[:3]
            wheel_image_alpha = wheel_image_rotated[3:]
            alpha_complement = 1 - wheel_image_alpha

            image = transform(image)
       
            for c in range(3):
                image[c, y_offset:y_offset+wheel_image_rotated.size(1), x_offset:x_offset+wheel_image_rotated.size(2)] = (
                image[c, y_offset:y_offset+wheel_image_rotated.size(1), x_offset:x_offset+wheel_image_rotated.size(2)] * alpha_complement
                + wheel_image_rgb[c] * wheel_image_alpha)
        
            image = transforms.ToPILImage()(image)

        if GraphMode == 3:
            # Show double WheelDrive
            x_offset_img = 0
            y_offset_img = 220
            x_offset = 80
            y_offset = 220

            angle_img = angle_img + angle0*2
            angle = angle + angle1*2
            rotate_transform = transforms.functional.rotate
            wheel_image_rotated_img = rotate_transform(wheel_image_resized, angle=angle_img)
            wheel_image_rotated = rotate_transform(wheel_image_resized, angle=angle)

            wheel_image_rgb_img = wheel_image_rotated_img[:3]
            wheel_image_alpha_img = wheel_image_rotated_img[3:]
            alpha_complement_img = 1 - wheel_image_alpha_img
            
            wheel_image_rgb = wheel_image_rotated[:3]
            wheel_image_alpha = wheel_image_rotated[3:]
            alpha_complement = 1 - wheel_image_alpha
            
            image = transform(image)
       
            for c in range(3):
                image[c, y_offset_img:y_offset_img+wheel_image_rotated_img.size(1), x_offset_img:x_offset_img+wheel_image_rotated_img.size(2)] = (
                image[c, y_offset_img:y_offset_img+wheel_image_rotated_img.size(1), x_offset_img:x_offset_img+wheel_image_rotated_img.size(2)] * alpha_complement_img
                + wheel_image_rgb_img[c] * wheel_image_alpha_img)
                                
                image[c, y_offset:y_offset+wheel_image_rotated.size(1), x_offset:x_offset+wheel_image_rotated.size(2)] = (
                image[c, y_offset:y_offset+wheel_image_rotated.size(1), x_offset:x_offset+wheel_image_rotated.size(2)] * alpha_complement
                + wheel_image_rgb[c] * wheel_image_alpha)
        
            image = transforms.ToPILImage()(image)
        
        if GraphMode == 5:
            # Faster R-CNN processing
            image3 = transform3(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                # New image
                output3 = model3(image3)
                boxes = output3[0]['boxes'].cpu().detach().numpy()
                scores = output3[0]['scores'].cpu().detach().numpy()
                labels = output3[0]['labels'].cpu().detach().numpy()
                # Filter out low-confidence predictions
                threshold = 0.5
                boxes = boxes[scores >= threshold]
                labels = labels[scores >= threshold]
                scores = scores[scores >= threshold]
                # Draw boxes
                #draw = ImageDraw.Draw(to_pil_image(image3.squeeze().cpu()))
                draw = ImageDraw.Draw(image)
                print("Number of boxes:",len(boxes))
                
                for box, label, score in zip(boxes, labels, scores):
                    print(f"Box: {box}, Label: {label}, Score: {score}")
                    x1, y1, x2, y2 = box
                    draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
                    draw.text((x1, y1), f"{label}: {score:.2f}", fill="red")
                
                #draw.rectangle(((50, 50), (100, 100)), outline="red", width=3)
                #draw.text((50, 50), "car", fill="red")
        # Update the Matplotlib plot
        #im_original.set_data(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        if GraphMode == 0 or GraphMode == 1 or GraphMode == 2 or GraphMode == 3:
            im_original.set_data(image)

        if GraphMode == 4:
            im_original1.set_data(image)
            im_mask1.set_data(output1_np)
            im_original2.set_data(image)
            im_mask2.set_data(output2_np)

        if GraphMode == 5:
            im_original.set_data(image)

        if GraphMode == 1 or GraphMode == 2:
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
        
        # Save the fig.canvas as a .jpg image
        #if frame_count != 0:  # Save the first frame as an example
        #    image_pil = Image.fromarray(cv2.cvtColor(plot_frame, cv2.COLOR_BGR2RGB))
        #    image_pil.save(str(frame_count)+"output_image.jpg")

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

