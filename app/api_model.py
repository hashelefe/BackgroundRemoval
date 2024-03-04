from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import cloudinary
from cloudinary.uploader import upload
import requests
import torch
import torchvision
import io


# Load DeepLabV3 model
model = torchvision.models.segmentation.deeplabv3_resnet101()
model.eval()

# Preprocessing transformation for images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def remove_background(image):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # If GPU available, move the input to GPU
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    output_predictions = output.argmax(0)

    # Create mask to keep only foreground
    mask = output_predictions.byte().cpu().numpy()
    masked_image = Image.fromarray((image * mask).astype('uint8'), 'RGB')
    
    return masked_image


async def remove_background_api(image_url: str):

    response = requests.get(image_url)
    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes))

    # Remove background
    modified_image = remove_background(image)

    # Upload modified image to Cloudinary
    cloudinary.config(
        cloud_name="dj0v7ypzd",
        api_key="889622549467972",
        api_secret="ZGDR6GYIHF53RsYJcZxmCl2nmnI"
    )
    upload_result = upload(modified_image, folder="cait")
    print(upload_result)
    # Return URL of the modified image
    return {"modified_image_url": upload_result['secure_url']}
