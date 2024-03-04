import numpy as np
import torch
import cloudinary
from cloudinary.uploader import upload
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import io
import cv2
import os  # For temporary file management


cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME')
api_key = os.environ.get('CLOUDINARY_API_KEY')
api_secret = os.environ.get('CLOUDINARY_API_SECRET')

def load_model():
    """
    Loads the DeepLabV3+ ResNet50 model with recommended weights.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights='DeepLabV3_ResNet50_Weights.DEFAULT')
    model.eval()
    return model

def make_transparent_foreground(pic, mask):
    """
    Creates a transparent foreground image from the original image and mask.

    Args:
        pic (PIL.Image): The original image.
        mask (np.ndarray): The binary foreground mask.

    Returns:
        PIL.Image: The transparent foreground image.
    """
    # Split the image into channels
    b, g, r = cv2.split(np.array(pic).astype('uint8'))

    # Add an alpha channel with all pixels initially set to transparent (255)
    a = np.ones(mask.shape, dtype='uint8') * 255

    # Merge the alpha channel back
    alpha_im = cv2.merge([b, g, r, a], 4)

    # Create a transparent background
    bg = np.zeros(alpha_im.shape)

    # Stack the mask to create a 4-channel mask for accurate copying
    new_mask = np.stack([mask, mask, mask, mask], axis=2)

    # Copy only foreground color pixels from the original image where the mask is set
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

    return foreground

def remove_background(model, input_bytes):
    """
    Removes background from image bytes using DeepLabV3+ ResNet50.

    Args:
        model (torch.nn.Module): The DeepLabV3+ ResNet50 model.
        input_bytes (bytes): The image bytes.

    Returns:
        tuple: (PIL.Image, np.ndarray) - The transparent foreground image and binary mask.
    """

    # Open image from bytes directly
    input_image = Image.open(io.BytesIO(input_bytes))

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

    # Create a binary mask of the foreground
    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)

    foreground = make_transparent_foreground(input_image, bin_mask)

    return foreground, bin_mask

def api_call(input_url: str):
    """
    Downloads an image from a URL, removes the background, and uploads it to Cloudinary.

    Args:
        input_url (str): The URL of the image.

    Returns:
        dict: {"modified_image_url": str, "error": str (optional)}
            - "modified_image_url": The URL of the modified image on Cloudinary (if successful).
            - "error": An error message if any errors occur during download or processing.

    Raises:
        OSError: If an error occurs during file operation.
        requests.exceptions.RequestException: If there's an error downloading the image.
    """
    response = requests.get(input_url)

    # Check and handle potential URL access errors gracefully
    if response.status_code != 200:
        return {"error": f"Failed to download image from URL: {input_url}"}

    image_bytes = response.content

    try:
        # Process image bytes directly
        deeplab_model = load_model()
        foreground, bin_mask = remove_background(deeplab_model, image_bytes)
        img_fg = Image.fromarray(foreground)
        img_fg = img_fg.convert('RGB')  # Ensure RGB format

        cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME')
        api_key = os.environ.get('CLOUDINARY_API_KEY')
        api_secret = os.environ.get('CLOUDINARY_API_SECRET')

        # Replace with your Cloudinary credentials
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret
        )

        # Correct indentation:
        img_bytes = io.BytesIO()
        img_fg.save(img_bytes, format='JPEG')  # Modify format if needed
        img_bytes.seek(0)

        upload_result = upload(img_bytes.read(), folder="cait")
        return {"modified_image_url": upload_result['secure_url']}
    except OSError as e:
        return {"error": f"Error processing image: {e}"}

# Example usage (replace with your actual Cloudinary credentials)
cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret
        )

input_url = "https://res.cloudinary.com/dj0v7ypzd/image/upload/v1709578479/cait/m0mlhgugzyorlstwkbeq.jpg"
response = api_call(input_url)

if "error" in response:
    print(f"Error: {response['error']}")
else:
    print(f"Modified image URL: {response['modified_image_url']}")
