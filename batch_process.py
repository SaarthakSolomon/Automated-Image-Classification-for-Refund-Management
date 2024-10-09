import torch
import os
from torchvision import transforms
from PIL import Image
import json
import schedule
import time
from app import load_trained_model, transform_image, class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # This should print 'cuda' if a GPU is available


# Function to process a batch of images in a directory
def process_batch(image_dir, output_file):
    print("Starting batch processing...")  # Debug statement

    # Load the trained model
    try:
        model = load_trained_model()
        print("Model loaded successfully.")  # Debug statement
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Ensure the output file directory exists
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"Output directory verified/created: {os.path.dirname(output_file)}")  # Debug statement
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    # List all image files in the directory
    try:
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"Found {len(image_files)} image(s) in the directory.")  # Debug statement
    except Exception as e:
        print(f"Error reading image directory: {e}")
        return

    if not image_files:
        print("No images found in the directory to process.")
        return

    results = []

    for image_file in image_files:
        try:
            # Load and transform the image
            image_path = os.path.join(image_dir, image_file)
            print(f"Processing image: {image_file}")  # Debug statement
            image = Image.open(image_path)
            image_tensor = transform_image(image).to(device)
            
            # Run the model to get the prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                category_index = predicted.item()
                category_name = class_names[category_index]
            
            # Append the result
            results.append({
                'image': image_file,
                'category': category_name
            })
            print(f"Processed image: {image_file} - Category: {category_name}")  # Debug statement

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")  # Debug statement
            results.append({
                'image': image_file,
                'error': str(e)
            })

    # Save the results to the output file
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Batch processing completed. Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

# Function to schedule the batch processing job
def scheduled_job():
    print("Scheduled job started.")  # Debug statement
    image_dir = 'D://Refund_Model//Data//batch'  # Directory containing the images for batch processing
    output_file = 'D://Refund_Model//Data//batch_results.json'  # Output file for results
    process_batch(image_dir, output_file)
    print("Scheduled job finished.")  # Debug statement

# Schedule the job to run every day at 1:00 AM
schedule.every().day.at("01:00").do(scheduled_job)

# Run the schedule
if __name__ == '__main__':
    print("Batch processing scheduled to run every day at 01:00 AM.")
    # Uncomment the line below if you want to test the job immediately
    #scheduled_job()
    while True:
        schedule.run_pending()
        time.sleep(60)
