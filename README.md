# Automated-Image-Classification-for-Refund-Management


## Overview
This project provides an automated solution for classifying refund items using a deep learning model based on the VGG16 architecture. The model is fine-tuned to classify images into 22 distinct categories, including electronics, clothing, and home decor. The project includes:
- A Flask API for deploying the model as a service.
- A batch processing system to automate classification on a scheduled basis.
- Data preprocessing techniques to enhance model robustness.

## Features
- **Automated Classification**: Classifies refund items into predefined categories using image data.
- **API Deployment**: Provides a REST API endpoint for real-time classification of uploaded images.
- **Batch Processing**: Scheduled batch classification for processing multiple images simultaneously.
- **Monitoring and Logging**: Integrated logging for tracking API usage and batch processing.

## Dataset
The dataset used in this project was sourced from [Kaggle](https://drive.google.com/file/d/1FTuwvVvNgbzHKEFY2EcVxK_D160J4GoC/view?usp=sharing)). It contains images of items labeled into 22 categories such as electronics, footwear, and accessories. Make sure to download and organize the dataset as described in the setup instructions.

## Installation
### Prerequisites
Ensure the following are installed on your system:
- Python 3.8 or above
- pip
- Git
- CUDA (if using GPU)

### Clone the Repository
```bash
git clone https://github.com/SaarthakSolomon/Automated-Image-Classification-for-Refund-Management.git
cd refund-item-classification
```

### Create a Virtual Environment
```bash
python -m venv refund_env
source refund_env/bin/activate # For Unix/macOS
refund_env\Scripts\activate # For Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Additional Requirements
Ensure you have CUDA installed if you intend to run the model on a GPU. Refer to [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads) for more details.

## Usage
### 1. Preparing the Dataset
- Download the dataset from [Here](https://drive.google.com/file/d/1FTuwvVvNgbzHKEFY2EcVxK_D160J4GoC/view?usp=sharing) and organize it as follows:
  ```
  Data/
      train/
          Category1/
              image1.jpg
              image2.jpg
          Category2/
              image1.jpg
              image2.jpg
      batch/
          image1.jpg
          image2.jpg
  ```
- Make sure to split your dataset into training and validation sets.

### 2. Training the Model
Run the `classification_model.py` script to train the model:
```bash
python classification_model.py
```
The trained model will be saved as `final_refund_item_classifier.pth`, and can be downloaded from ([final_refund_item_classifier.pth](https://drive.google.com/file/d/1FTuwvVvNgbzHKEFY2EcVxK_D160J4GoC/view?usp=sharing)).

### 3. Running the Flask API
Start the Flask API by running:
```bash
python app.py
```
The API will be accessible at `http://127.0.0.1:5000`.

#### API Endpoints
- **GET /**: Home endpoint displaying information about the API.
- **POST /predict**: Accepts images for classification. Use a tool like Postman or cURL to test:
    - **Postman**:
        - Method: POST
        - URL: `http://127.0.0.1:5000/predict`
        - Body: Select `form-data`, key as `images` (file type), and upload images.
    - **cURL**:
        ```bash
        curl -F "images=@/path/to/image.jpg" http://127.0.0.1:5000/predict
        ```

### 4. Batch Processing
The batch processing job is scheduled to run daily. You can manually run the batch process using:
```bash
python batch_process.py
```
Ensure that the batch directory (`Data/batch`) contains images for processing. The results will be saved in `batch_results.json`.

### 5. Monitoring and Logging
- Logs for API calls and batch processing are stored in the `logs` directory.
- Use tools like Prometheus or Grafana (recommended) to monitor API usage and system performance.

## Project Structure
```
refund-item-classification/
│
├── Data/
│   ├── train/
│   ├── val/
│   └── batch/
│
├── app.py               # Flask API implementation
├── batch_process.py     # Script for scheduled batch processing
├── classification_model.py  # Model training script
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── final_refund_item_classifier.pth # Trained model weights
```

## Troubleshooting
- **Model Loading Errors**: Ensure that the path to `final_refund_item_classifier.pth` is correct in `app.py`.
- **CUDA Errors**: Verify that CUDA is installed correctly and your environment is configured to use GPU.
- **API Testing Issues**: Make sure the API is running at the correct address (`http://127.0.0.1:5000`) and that Postman/cURL configurations are correct.

## Future Improvements
- **Model Enhancements**: Experiment with other architectures like ResNet for improved accuracy.
- **API Optimization**: Implement caching mechanisms for faster inference.
- **Scalability**: Deploy the API on cloud platforms (e.g., AWS, Azure) for handling large-scale requests.

## References
- **Dataset**: [Dataset URL](https://drive.google.com/file/d/1FTuwvVvNgbzHKEFY2EcVxK_D160J4GoC/view?usp=sharing)
- **VGG16 Model**: [PyTorch VGG16 Documentation](https://pytorch.org/vision/stable/models.html)
- **Python Libraries**: torch, torchvision, Flask, PIL, schedule
- **CUDA Setup**: [NVIDIA CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)

---
