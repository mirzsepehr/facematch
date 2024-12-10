from fastapi import (FastAPI, 
                     File, 
                     UploadFile, 
                     HTTPException, 
                     Request)
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision import transforms
import requests
from io import BytesIO


app = FastAPI()
allowed_image_types = ["image/jpeg", "image/png", "image/gif"]
resnet = InceptionResnetV1(pretrained="vggface2").eval()
face_detector_data = cv2.FaceDetectorYN.create(
                    model=r"/facematch/app/yunet.onnx",
                    config="",
                    input_size=(320, 240),
                    score_threshold=0.9, 
                    nms_threshold=0.3,
                    top_k=5000
                )
face_detector_anchor = cv2.FaceDetectorYN.create(
                    model=r"/facematch/app/yunet.onnx",
                    config="",
                    input_size=(240, 320),
                    score_threshold=0.9, 
                    nms_threshold=0.3,
                    top_k=5000
                )

class notReceivedException(Exception):
    def __init__(self, name: str):
        self.name = name
@app.exception_handler(notReceivedException)
async def notReceived_exception_handler(request: Request, exc:notReceivedException):
    return JSONResponse(
        status_code=404,
        content={"message": f"Oops! {exc.name} didn't receive!"},
    )



async def grayscale_to_3channels(image):
    """
    Converts a grayscale PIL Image 
    to a 3-channel image. 
    """
    
    if image.mode != 'L':
        raise ValueError("Image must be in grayscale mode (L)")

    return Image.merge("RGB", (image, image, image))

async def convert2cv2(pil_image):
    """
    converts PIL images
    to opencv images
    """
    image = pil_image.convert('RGB')
    opencvImage=np.array(image)
    if opencvImage is None or np.all(opencvImage==0):
        return np.zeros(shape=(150, 150, 3))
    opencvImage=cv2.cvtColor(opencvImage, cv2.COLOR_BGR2RGB)
    return opencvImage

async def convert2PIL(cv2_image):
    """
    converts opencv image 
    to PIL image.
    """
    if cv2_image is None:
        raise ValueError("File is empty.")
    
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # opencvImage = cv2_image[:, :, ::-1].copy()
    pil_img = Image.fromarray(img)

    # For reversing the operation:
    # pil_img = np.asarray(pil_img)
    return pil_img

async def rotate_180(image_array):
    """
    rotate the image
    180 degrees.
    """
    return np.rot90(np.rot90(image_array))

async def normalize_illumination(image):
    """
    Normalizes the illumination 
    of an image using PIL.
    """

    img_array = np.array(image)
    mean = np.mean(img_array)
    std = np.std(img_array)
    img_array = (img_array - mean) / std
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255

    normalized_image = Image.fromarray(img_array.astype('uint8'))

    return normalized_image

async def resize_image(img, target_size):
    """
    Resizes an image to 
    the specified target size.
    Target size should be like this-> e.g: (200, 200)
    """
    resizedImage=cv2.resize(img, target_size, cv2.INTER_AREA)
    return resizedImage

async def face_extraction(faces_list, img):
    """
    This function withdraws faces from the image
    to send as an input to our deep learning model.
    """
    if faces_list[1] is not None:
        for face in faces_list[1]:
            x, y, w, h = face[0:4].astype(int)
            normalized_image = cv2.normalize(
                img, 
                None, 
                alpha=0, 
                beta=255, 
                norm_type=cv2.NORM_MINMAX)
            cv2.rectangle(normalized_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            #h>25 and w>20 is because we only are interested in detecting driver's face which is closer to the camera.
            if h>25 and w>20:
                #licence pictures shouldn't be normalized 
                roi_color = normalized_image[y:y + h, x:x + w] 
                try:
                    resized_image=await resize_image(roi_color, (150, 150))
                    # cv2.imwrite("output_extr.jpg",resized_image)
                    return resized_image
                except Exception as e:
                    # print("NO FACE DETECTED")
                    print(e)
            else:
                # print(f"NO FACE DETECTED")
                return np.zeros(shape=(150, 150, 3))
        else:
            return np.zeros(shape=(150, 150, 3))
            
async def load_facematch_model():
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return resnet

async def predict(lisence_image:Image.Image, image:Image.Image):
    """
    This function preprocesses the extracted images
    and then gives preprocessed images to the
    Resnet deep learning model. the output is
    the 'distance'(distance of embeddings of neural network which is float value),
    and 'prediction'(true or false, the threshold to have true output is distance<0.9).
    I proceed to this number with trial and error.
    you can exert some changes to this threshold number4 if needed.  
    """
    transform = transforms.ToTensor()

    #image anchor is the licence image
    image_anchor = ImageEnhance.Contrast(lisence_image)
    image_anchor = image_anchor.enhance(1)
    tensor_imgAnchor = transform(image_anchor)

    #convert images to opencv object
    numpy_image_data = np.array(image)
    opencv_image_data = cv2.cvtColor(numpy_image_data, cv2.COLOR_RGB2BGR)

    #histogram equalization
    hist,_ = np.histogram(opencv_image_data.flatten(),256,[0,256])
    cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img_hist = cdf[image]

    #convert back to PIL images
    pil_image = Image.fromarray(img_hist)

    #preprocessing images
    image_data_gryscale = ImageOps.grayscale(pil_image)
    normalized_image = await normalize_illumination(image_data_gryscale)
    denoised_image_data = normalized_image.filter(ImageFilter.MedianFilter(size=3))
    
    #Get embeddings of my model
    tensor_imgData = transform(await grayscale_to_3channels(denoised_image_data))

    embeddings_anchor = resnet(tensor_imgAnchor.unsqueeze(0)).detach()
    embeddings_data = resnet(tensor_imgData.unsqueeze(0)).detach()

    #The threshold needed to get true results for facematch is considered 0.9
    distance = (embeddings_anchor - embeddings_data).norm().item()
    return distance, distance<0.9


# @app.get("/")
# async def get_root():
#     print("Hi, Im' in the root!")
#     return {"message":"Hey there, this is root."}

@app.post("/facematch")
async def facematch_api(image_anchor:UploadFile=File(...), image_data:UploadFile=File(...)):
    #Raise error if input files are Null:
    if not image_anchor or not image_data:
        raise notReceived_exception_handler(name = "file")
    
    #image data types which are mentioned above are allowed data types:
    if image_anchor.content_type not in allowed_image_types or image_data.content_type not in allowed_image_types:
        raise HTTPException(status_code=400, 
                            detail="Invalid File type. (only jpeg, png or gif are allowed)"
                        )
    contents_data = await image_data.read() 
    contents_anch = await image_anchor.read() 
    pil_image_data = Image.open(BytesIO(contents_data))
    pil_image_anchor = Image.open(BytesIO(contents_anch))

    #Some preprocessing on images:
    pil_image_data = pil_image_data.rotate(180)
    
    image_data = await convert2cv2(pil_image_data)
    image_anchor = await convert2cv2(pil_image_anchor)
    
    faces_data = face_detector_data.detect(image_data)
    faces_anchor = face_detector_anchor.detect(image_anchor)

    extracted_anchor_face = await face_extraction(faces_anchor, image_anchor)
    extracted_data_face = await face_extraction(faces_data, image_data)

    #if face could not be detected in the image, no face will be extracted. 'face_extraction' function will return np.ndarray of zeros.
    if np.all(extracted_data_face==0) or extracted_data_face is None:
        return HTTPException(status_code=400, detail="No Face Detected in the Picture.(the variable 'exreacted_data_face' is None)")
    extracted_data_face = await convert2PIL(extracted_data_face)
    extracted_anchor_face = await convert2PIL(extracted_anchor_face)
    _, prediction = await predict(extracted_anchor_face, extracted_data_face)
    return {"pred":prediction}