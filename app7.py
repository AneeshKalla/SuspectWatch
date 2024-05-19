from flask import Flask, request, jsonify, render_template, send_file, abort
import cv2  # Import OpenCV
import os
import shutil
from ultralytics import YOLO
from datetime import datetime
import numpy as np
from PIL import Image
import tqdm
import logging
logging.basicConfig(level=logging.DEBUG)
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load CLIP model and processor


app = Flask(__name__)
app.debug = True

print("worked")

model = YOLO('yolov8m-seg.pt')

video_directory = "FrontEndVideo"

# Ensure the video directory exists
if not os.path.exists(video_directory):
    os.makedirs(video_directory)

@app.route('/')
def index():
    return render_template('index.html')


test_images_dir = 'test2017'
output_dir = 'yolo_preds'  # To save annotated images in the same directory
laptop_output_dir = 'people'

# Ensure output_dir exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def resize_mask(mask, frame_shape):
    """
    Resize the mask to match the frame's shape.
   
    :param mask: The mask to resize.
    :param frame_shape: The shape of the frame.
    :return: Resized mask.
    """
    # Resize the mask to match the frame's dimensions
    return cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)


@app.route('/process-input', methods=['POST'])
def find_frames():
    print("ACTUALLyy! STARTED QUERYING")
    #return jsonify(["1.png"])
    data = request.get_json()
    text = data.get('description')
    text_embedding = generate_text_embedding(text, embedding_model, processor)
    queries = query_embeddings(vector_db, text_embedding, 3)
    indices = queries[1]
    print(indices)
    print(type(indices))
    frame_list = []
    for i in indices[0]:
        filename = filenames[i]
        shutil.copyfile(filename, os.path.join("thief_frames", os.path.basename(filename)))
        frame_list.append(os.path.join("thief_frames", os.path.basename(filename)))

    print(frame_list)
    #return jsonify(["1.png"])
    return jsonify(frame_list)

@app.route('/test', methods=['GET'])
def test():
    print("STARTED QUERYING")
    return()


def segment_and_annotate_images(frame, time, batch_size=1):
    # List all images in the testImages directory
   
    for i in range(1):
       
        frames = [frame]

        # Make predictions on the batch of frames
        preds = model.predict(frames, batch=batch_size)  # This should return a list of predictions

        for frame, pred in zip(frames, preds):
           
           

            # Make predictions on the frame
            #pred = model.predict(frame)[0]  # Assume this returns a list of predictions with masks
            frame_with_overlay = pred.plot()
            # Here you would process your predictions and overlay them on the frame
            # For the sake of example, let's assume you have a function that does this:
            # frame_with_overlay = apply_predictions_to_frame(frame, pred)
           
            # Save the annotated image
            annotated_img_name = f"annotated_{time}.png"
            annotated_img_path = os.path.join(output_dir, annotated_img_name)
            cv2.imwrite(annotated_img_path, frame_with_overlay)
           
            #print(f"masks: {pred[0].masks}")

            #print(type(pred))
            #print(pred)
       
            # Now, segment out each laptop if the predictions include bounding boxes or masks for laptops
            laptopCount = 0
            laptop_class_id = 0
            for detection in pred:
                if detection.boxes and detection.masks:  # Ensure there are boxes and masks
                    for i, box in enumerate(detection.boxes):
                        #print(box.cls)
                        if box.cls.item() == laptop_class_id:  # Check if the detected class is a laptop
                            mask = detection.masks.data[i]  # Get the corresponding mask
                            resized_mask = resize_mask(mask.cpu().numpy(), frame.shape)

                            print(f"item: {box.cls.item()}")
                            mask_bool = resized_mask.astype(bool)

                            # Find bounds for cropping using the mask
                            rows = np.any(mask_bool, axis=1)
                            cols = np.any(mask_bool, axis=0)
                            rows_nonzero = np.where(rows)[0]
                            cols_nonzero = np.where(cols)[0]

                            if rows_nonzero.size and cols_nonzero.size:
                                y_min, y_max = rows_nonzero[[0, -1]]
                                x_min, x_max = cols_nonzero[[0, -1]]

                                bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
                                # Calculate the area of the frame
                                frame_area = frame.shape[0] * frame.shape[1]
                                if bbox_area / frame_area <= 0.1:
                                    continue

                                # Crop the original image (frame) to these bounds
                                cropped_frame = frame[y_min:y_max+1, x_min:x_max+1]
                                cropped_mask_bool = mask_bool[y_min:y_max+1, x_min:x_max+1]

                                # Create an empty black image with the same shape as the cropped frame
                                laptop_img_cropped = np.zeros_like(cropped_frame)

                                # Copy the pixels of the cropped frame where the mask is True
                                laptop_img_cropped[cropped_mask_bool] = cropped_frame[cropped_mask_bool]
                            else:
                                continue

                            # laptop_img_cropped now contains the zoomed-in part of the image
                            # Save or process the laptop_img as needed
                            # For example, to save:
                            laptop_img_name = f"{laptopCount}_{time}.png"
                            laptopCount+=1
                            laptop_img_path = f"{laptop_output_dir}/{laptop_img_name}"
                            path_with_png = laptop_img_path[0:len(laptop_img_path)-3] + "png"
                            cv2.imwrite(path_with_png, laptop_img_cropped)
                            image_embedding = generate_image_embedding(path_with_png, embedding_model, processor)
                            #print(image_embedding)
                            add_embedding_to_index(vector_db, path_with_png, image_embedding)
                            print(f"amt embeddings: {vector_db.ntotal}")

# Execute the function



# Process the images


last_seconds = 0
results = []

@app.route('/upload-video', methods=['POST'])
def upload_video():
    global last_seconds  # Indicate that we'll be using the global variable

    video = request.files['video']
    if video:
        # Save the video clip to the FrontEndVideo directory
        video_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.mov"
        video_path = os.path.join(video_directory, video_filename)
        video.save(video_path)

        # Process the video and extract frames
        process_video_with_yolo(video_path)

        print(f"Video uploaded and saved as {video_path}")
        return jsonify({"message": "Video uploaded and processed successfully"}), 200
    else:
        print("No video uploaded")
        return jsonify({"error": "No video uploaded"}), 400

def process_video_with_yolo(video_path):
    global last_seconds
    print("Starting YOLO processing")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    interval_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        interval_counter += 1
        if interval_counter == 15:  # Process approximately 1 frame every 3 seconds (30 frames per second * 3 seconds)
            interval_counter = 0
            time = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            segment_and_annotate_images(frame, time)
            #pred = model(frame)
            #frame_with_overlay = pred[0].plot()
            #results.append((frame, frame_with_overlay, pred))

            #timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            #seconds = frame_count // 30
            #filename = f"generalFrames/annotated_{seconds}seconds.png"
            #cv2.imwrite(filename, frame_with_overlay)

    cap.release()

@app.route('/last-overlayed-image', methods=['GET'])
def last_overlayed_image():
    if results:
        last_result = results[-1]  # Get the last element in results
        last_image = last_result[1]
        pil_image = Image.fromarray(last_image.astype('uint8'))
        filename = 'last_image.png'
        pil_image.save(filename)
        return send_file(filename, mimetype='image/png')
    else:
        return abort(404, description="No overlaid image found.")
   
from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor
embedding_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def generate_image_embedding(image_path, embedding_model, processor, device="cpu"):
    embedding_model.eval()
    with torch.no_grad():
        # Convert numpy array to PIL image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        # Move inputs to the specified device
        pixel_values = inputs["pixel_values"].to(device)

        # Generate image embedding
        outputs = embedding_model.get_image_features(pixel_values=pixel_values)

        # Return the embedding as a numpy array
        image_embedding = outputs.cpu().numpy()
   
    return image_embedding
   
def generate_text_embedding(text, embedding_model, processor, device="cpu"):
    embedding_model.eval()
    with torch.no_grad():
        # Preprocess the text
        inputs = processor(text=text, return_tensors="pt")

        # Move inputs to the specified device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate text embedding
        outputs = embedding_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        # Return the embedding as a numpy array
        text_embedding = outputs.cpu().numpy()

    return text_embedding


import faiss
import numpy as np


def initialize_index(embedding_dim):
    vector_db = faiss.IndexFlatL2(embedding_dim)
    return vector_db

# Example usage:
embedding_dim = 512  # Example embedding dimension
vector_db = initialize_index(embedding_dim)


filenames = []
def add_embedding_to_index(vector_db, filename,embedding):
    vector_db.add(embedding)
    filenames.append(filename)


def query_embeddings(vector_db, query_embedding,  k=5):
    distances, indices = vector_db.search(query_embedding, k)
    return distances, indices


def delete_and_create_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print(f"Deleted the directory: {directory_path}")
    os.makedirs(directory_path)
    print(f"Created a new directory: {directory_path}")

delete_and_create_directory("thief_frames")
delete_and_create_directory("people")

    



def process_all_images_in_directory(directory):
    """
    Process all images in the specified directory.
   
    :param directory: The directory containing images to process.
    
    """
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust the extensions as needed
            image_embedding = generate_image_embedding(os.path.join(directory, filename), embedding_model, processor)
            print(len(image_embedding[0]))

        else:
            print("Did not work")
            break

#process_all_images_in_directory(output_dir)

#def description_of_code():
    #text = input
    #text_embedding = generate_text_embedding(text, model, processor)
    #print(text_embedding)

   



if __name__ == '__main__':
    app.run(debug=True)