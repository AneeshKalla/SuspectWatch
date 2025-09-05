https://github.com/user-attachments/assets/c5929fee-d531-4e1b-966d-5045e60b5471


# SuspectWatch

Inspired by a real-life experience where our basketball was stolen from a neighborhood court, SuspectWatch aims to help real detectives solve mysteries in big cities. In places where organized crime and repeat offenders are common, SuspectWatch provides a software solution to help identify and catch suspects using computer vision and natural language processing.

## Project Overview

SuspectWatch uses advanced AI models to match descriptions of suspects to individuals captured in CCTV footage. The system works by:

- **Extracting people from video frames** using YOLOv8 object detection.
- **Removing backgrounds** to focus on the person for identification.
- **Generating text descriptions** for each detected person using GPT-4o.
- **Training a CLIP model** (Contrastive Language-Image Pretraining) to create embeddings for both images and text descriptions.
- **Matching suspects** by comparing the embeddings of people in footage to the embedding of a provided text description.

## How It Works

1. **Dataset Preparation**
	- The COCO dataset is processed with YOLOv8 to extract images of people.
	- GPT-4o generates a text description for each person detected.
	- These image-text pairs are used to contrastively train CLIP, aligning visual and textual representations.

2. **Suspect Identification Workflow**
	- A website interface allows users to upload video footage.
	- YOLOv8 runs on video frames to detect people.
	- The vision transformer creates embeddings for each detected person.
	- The text transformer creates an embedding for the suspect description.
	- The system compares embeddings and displays the closest matches from the footage.

## Detailed Workflow

- Use COCO dataset and YOLOv8 to extract images of people.
- Use GPT-4o to generate text descriptions for each image.
- Contrastively train CLIP (vision and text transformer) to match images and descriptions.
- Website takes video footage, runs YOLOv8 on frames, runs image transformer on people, runs text transformer on thief description, and shows closest matches from footage.

## Repository Structure

- `app7.py`: Main application logic.
- `filter_data.ipynb`, `ml.ipynb`, `ml_v2.ipynb`, etc.: Jupyter notebooks for data processing and model training.
- `coco_people_v0/`, `coco_people_v1/`: Extracted images of people from COCO.
- `image_descriptions.csv`: Text descriptions generated for each image.
- `yolov8m-seg.pt`: YOLOv8 model weights.
- `templates/`, `FrontEndVideo/`: Website and frontend code.
- Other folders: Results, logs, and additional data.

## Getting Started

1. Clone the repository.
2. Install dependencies (YOLOv8, CLIP, GPT-4o API access).
3. Run the notebooks to process data and train models.
4. Launch the web interface to upload footage and search for suspects.

## License

This project is for educational and research purposes.

