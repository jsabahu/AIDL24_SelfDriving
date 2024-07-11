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
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor,FasterRCNN_ResNet50_FPN_Weights)
import torchvision
import random

#GraphMode = 0 # Just Video with WheelDrive
#GraphMode = 1 # Compare models
#GraphMode = 2 # All
GraphMode = 3 # Just Video with WheelDrive (Image vs Model) 
#GraphMode = 4 # Compare models overwriting lines
#GraphMode = 5 # Just Video with WheelDrive and Car Detection
format = "video" # "image" not supports modes / "video" supports all modes

if format == "image" or format == "video":
    print("Creating a",format, "in mode",str(GraphMode))
else:
    print("Wrong conditions ( format =",format,"/ mode =",str(GraphMode),")")
    raise SystemExit  # Exit by wrong conditions

#video_file = 'data/Example1.mp4'
video_file = 'data/ObjectDetection1.mp4'
#video_file = 'data/sample2.mp4'
wheel_file = 'data/drivewheel.png'
image_file1 = 'data/imagen.png'
image_file2 = 'data/b0a1dce9-1135e8fe.jpg'

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
# Function to set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Set seeds for reproducibility
set_seed(10)
# Load the pre-trained Faster R-CNN model
model3 = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

# Replace the pre-trained head with a new one
model3.roi_heads.box_predictor = FastRCNNPredictor(
    model3.roi_heads.box_predictor.cls_score.in_features,
    config["hyperparameters"]["num_classes"]
    )
# Load the trained model weights
state_dict = torch.load('models/Faster_R_CNN.pth', map_location=DEVICE)
model3.load_state_dict(state_dict, strict=False)

# Set the model to evaluation mode
model3.eval()

# Transform for Faster R-CNN
transform3 = FCnn.create_transforms(config)

if format == "video":
    # Open the video file
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
    if GraphMode == 0 or GraphMode == 5:
        output_video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (2*target_size[1], 2*target_size[0]))
    if GraphMode == 1 or GraphMode == 2 or GraphMode == 3:
        output_video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (2*target_size[1]*3, 2*target_size[0]))
    if GraphMode == 4:
        output_video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (2*target_size[1]*2, 2*target_size[0]))


    # Initialize angle
    angle = 0
    angle_img = 0

    # Create the plot window
    if GraphMode == 0:
        fig, axes = plt.subplots(1, 1, figsize=(2*target_size[1]*0.01, 2*target_size[0]*0.01))
        im_original = axes.imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
        axes.set_title("Original Frame")
        axes.axis('off')

    if GraphMode == 1:
        fig, axes = plt.subplots(1, 3, figsize=(2*target_size[1]*0.03, 2*target_size[0]*0.01))
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
        fig, axes = plt.subplots(1, 3, figsize=(2*target_size[1]*0.03, 2*target_size[0]*0.01))
        im_original = axes[0].imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
        axes[0].set_title("Original Frame")
        axes[0].axis('off')
        annotation0 = axes[0].text(160, 20, "0", ha='center', va='center', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        im_predicted_image1 = axes[1].imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
        axes[1].set_title("Mask R-CNN Frame")
        axes[1].axis('off')
        annotation1 = axes[1].text(160, 20, "0", ha='center', va='center', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        im_predicted_image2 = axes[2].imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
        axes[2].set_title("LaneNET Frame")
        axes[2].axis('off')
        annotation2 = axes[2].text(160, 20, "0", ha='center', va='center', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    if GraphMode == 4:
        cmap_colour = LinearSegmentedColormap.from_list('colour', [(0, 'red'), (1, 'white')])
        fig, axes = plt.subplots(1, 2, figsize=(2*target_size[1]*0.02, 2*target_size[0]*0.01))
        im_original1 = axes[0].imshow(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
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
                angle_r1 = 0
                angle_r2 = 0

            image_prev = image
        
            if GraphMode == 2:
                annotation1.set_text(f"Rotation: {angle1:.2f} deg")
                annotation2.set_text(f"Rotation: {angle2:.2f} deg")

            if GraphMode == 3:
                annotation0.set_text(f"Image: {angle0:.2f} deg")
                annotation1.set_text(f"Rotation: {angle1:.2f} deg")
                annotation2.set_text(f"Rotation: {angle2:.2f} deg")

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
                # Show three WheelDrive
                x_offset_img = 40
                y_offset_img = 220
                x_offset1 = 40
                y_offset1 = 220
                x_offset2 = 40
                y_offset2 = 220

                angle_img += angle0*2
                angle_r1 += angle1*2
                angle_r2 += angle2*2
                rotate_transform = transforms.functional.rotate

                wheel_image_rotated_img = rotate_transform(wheel_image_resized, angle=angle_img)
                wheel_image_rotated1 = rotate_transform(wheel_image_resized, angle=angle_r1)
                wheel_image_rotated2 = rotate_transform(wheel_image_resized, angle=angle_r2)

                wheel_image_rgb_img = wheel_image_rotated_img[:3]
                wheel_image_alpha_img = wheel_image_rotated_img[3:]
                alpha_complement_img = 1 - wheel_image_alpha_img
            
                wheel_image_rgb1 = wheel_image_rotated1[:3]
                wheel_image_alpha1 = wheel_image_rotated1[3:]
                alpha_complement1 = 1 - wheel_image_alpha1
            
                wheel_image_rgb2 = wheel_image_rotated2[:3]
                wheel_image_alpha2 = wheel_image_rotated2[3:]
                alpha_complement2 = 1 - wheel_image_alpha2
 
                image1 = transform(image)
                image2 = transform(image)
                image = transform(image)
       
                for c in range(3):
                    image[c, y_offset_img:y_offset_img+wheel_image_rotated_img.size(1), x_offset_img:x_offset_img+wheel_image_rotated_img.size(2)] = (
                    image[c, y_offset_img:y_offset_img+wheel_image_rotated_img.size(1), x_offset_img:x_offset_img+wheel_image_rotated_img.size(2)] * alpha_complement_img
                    + wheel_image_rgb_img[c] * wheel_image_alpha_img)
                    
                    image1[c, y_offset1:y_offset1+wheel_image_rotated1.size(1), x_offset1:x_offset1+wheel_image_rotated1.size(2)] = (
                    image1[c, y_offset1:y_offset1+wheel_image_rotated1.size(1), x_offset1:x_offset1+wheel_image_rotated1.size(2)] * alpha_complement1
                    + wheel_image_rgb1[c] * wheel_image_alpha1)
                    
                    image2[c, y_offset2:y_offset2+wheel_image_rotated2.size(1), x_offset2:x_offset2+wheel_image_rotated2.size(2)] = (
                    image2[c, y_offset2:y_offset2+wheel_image_rotated2.size(1), x_offset2:x_offset2+wheel_image_rotated2.size(2)] * alpha_complement2
                    + wheel_image_rgb2[c] * wheel_image_alpha2)
                    
                image = transforms.ToPILImage()(image)
                image1 = transforms.ToPILImage()(image1)
                image2 = transforms.ToPILImage()(image2)

            if GraphMode == 5:
                # Faster R-CNN processing
                resize_gain_width = image.size[0]/config["transforms"]["resize_width"]
                resize_gain_height = image.size[1]/config["transforms"]["resize_height"]

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
                        x1, y1, x2, y2 = box
                        x1 = x1 * resize_gain_width
                        y1 = y1 * resize_gain_height
                        x2 = x2 * resize_gain_width
                        y2 = y2 * resize_gain_height
                        if (abs(x1 - x2)>=10) and (abs(y1 - y2)>=10): 
                            draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
                            draw.text((x1, y1), f"{label}: {score:.2f}", fill="red")
                            print(f"Valid Box: {box}, Label: {label}, Score: {score}")
                        else:
                            print(f"Discard Box: {box}, Label: {label}, Score: {score}")
                
                    #draw.rectangle(((50, 50), (100, 100)), outline="red", width=3)
                    #draw.text((50, 50), "car", fill="red")
            # Update the Matplotlib plot
            #im_original.set_data(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            if GraphMode == 0 or GraphMode == 1 or GraphMode == 2:
                im_original.set_data(image)

            if GraphMode == 3:
                im_original.set_data(image)
                im_predicted_image1.set_data(image1)
                im_predicted_image2.set_data(image2)

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
        
        frame_count += 1

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    output_video.release()

if format == "image":
    image_from_file1 = Image.open(image_file1).convert("RGB")   # Image for validation using plot blocked
    image_from_file2 = Image.open(image_file2).convert("RGB")   # Image for validation using plot blocked

    # Mask R-CNN processing
    image1 = transform1(image_from_file1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # Process image
        output1 = model1(image1, generate_full_image_rois(1, target_size, 0))
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=DEVICE).view(1, 3, 1, 1)
        output1 = (output1 * weights).sum(dim=1, keepdim=True)
        output1 = F.interpolate(output1, size=target_size, mode="bilinear", align_corners=False)
        output1 = (output1 - output1.min()) / (output1.max() - output1.min())
        output1 = (output1 > 0.001).type(torch.int)
        output1_np = output1.detach().cpu().numpy()[0].transpose(1, 2, 0)
        
    # LaneNet processing
    image2 = transform2(image_from_file1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # Process image
        output2 = model2(image2)
        output2 = torch.squeeze(output2["instance_seg_logits"],dim=0).transpose(1, 2).transpose(0, 2)
        weights2 = torch.tensor([0.2989, 0.5870, 0.1140], device=DEVICE).view(1, 1, 3)
        output2 = (output2 * weights2).sum(dim=2, keepdim=True)
        output2 = (output2 - output2.min()) / (output2.max() - output2.min())
        output2 = (output2 > 0.5).type(torch.int)
        output2_np = output2.detach().cpu().numpy()

    # Faster R-CNN processing
    image3 = transform3(image_from_file2).unsqueeze(0).to(DEVICE)
    with torch.no_grad():

        # Process image
        resize_gain_width = image_from_file2.size[0]/config["transforms"]["resize_width"]
        resize_gain_height = image_from_file2.size[1]/config["transforms"]["resize_height"]
        
        print("width gain: ",resize_gain_width,"heigh gain:",resize_gain_height)
        resize_gain_width = 1.6
        resize_gain_height = 0.9
        
        output3 = model3(image3)
        boxes = output3[0]['boxes'].cpu().detach().numpy()
        scores = output3[0]['scores'].cpu().detach().numpy()
        labels = output3[0]['labels'].cpu().detach().numpy()
        # Filter out low-confidence predictions
        threshold = 0.1
        boxes = boxes[scores >= threshold]
        labels = labels[scores >= threshold]
        scores = scores[scores >= threshold]

        # Draw boxes
        #draw = ImageDraw.Draw(to_pil_image(image3.squeeze().cpu()))
        draw = ImageDraw.Draw(image_from_file2)
        
        # Trained boxes for image 'data/b0a1dce9-1135e8fe.jpg'
        """ 
        boxes = [(469.356705,275.062311, 663.419073,420.187209),
		(501.41918,242.41960500349512,585.794122,300.374797),
		(578.3656470991442,248.062329,608.2145117020921,285.1444504155577),
		(636.419088,244.687333,694.2494743811769,289.8266252552358),
		(604.7028805723335,240.66378943861585,643.3308229996777,271.6831977514832),
		(1046.481308,281.812306,1279.356149,582.1767379223847),
		(191.383896571842,230.59711085414602,379.2561620139254,296.999798),
		(152.17068228953798,226.1490456495058,208.35678036567506,267.11807549668913),
		(324.231803,226.124845,374.856767,249.749826),
		(378.0856183040059,229.54362419438039,397.399589517678,250.61341097293177),
		(416.7135607313501,231.187341,447.419221,250.61341097293177),
		(1138.3624286102763,273.570906582146,1200.3237292618307,310.4339588685139),
		(596.5090746028968,240.66378943861585,613.4819583967299,251.78395468285132),
		(465.7593439630861,232.1188167839874,474.53842178748255,249.67697243278025),
		(494.43766485611445,234.45990420382645,519.6043546193841,252.0180598526193),
		(454.0539068638909,229.77772936414837,466.3446158180459,245.58006944806192),
		(400.3259487924768,231.29943975925966,415.5430170214306,249.44286726301226),
		(413.2019296015916,230.7141679042999,424.90736670078684,243.00487685845488)
		]
        labels = [(1),(1),(1),(1),(1),(1),(1),(1),(1),(1),(1),(1),(1),(1),(1),(1),(1),(1)]
        scores = [(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75),(0.75)]
        """
        print("Number of boxes:",len(boxes))
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            x1 = x1 * resize_gain_width
            y1 = y1 * resize_gain_height
            x2 = x2 * resize_gain_width
            y2 = y2 * resize_gain_height

            if (abs(x1 - x2)>=10) and (abs(y1 - y2)>=10): 
                draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
                draw.text((x1, y1), f"{label}: {score:.2f}", fill="red")
                print(f"Valid Box: {box}, Label: {label}, Score: {score}")
            else:
                print(f"Discard Box: {box}, Label: {label}, Score: {score}")

    cmap_colour = LinearSegmentedColormap.from_list('colour', [(0, 'red'), (1, 'white')])
    fig, axes = plt.subplots(1, 3, figsize=(2*target_size[1]*0.03, 2*target_size[0]*0.01))

    transf = transforms.Resize((target_size[0], target_size[1]))
    axes[0].imshow(transf(image_from_file1))
    axes[0].imshow(output1_np, cmap=cmap_colour, vmin=0, vmax=1, alpha=0.2)
    axes[0].set_title("Mask rCNN")
    axes[0].axis('off')

    transf = transforms.Resize((256, 512))
    axes[1].imshow(transf(image_from_file1))
    axes[1].imshow(output2_np, cmap=cmap_colour, vmin=0, vmax=1, alpha=0.2)
    axes[1].set_title("LaneNet")
    axes[1].axis('off')

    axes[2].imshow(image_from_file2)
    axes[2].set_title("Faster rCNN")
    axes[2].axis('off')

    # Show the plot
    plt.show()