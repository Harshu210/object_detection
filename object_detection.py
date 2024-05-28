# INSTALL LIBRARIES
pip install imageai
pip install tensorflow

# IMPORT LIBRARIES AND MODULES
from imageai.Detection import ObjectDetection
from keras.models import load_model
from IPython.display import Image

# CREATE A DIRECTORY AND STORY INPUT AND OUT FILES ALONG WITH THE MOIDEL FILE
!cd /content/
!mkdir /content/Input/
!mkdir /content/Output/
!mkdir /content/Model

# LOAD YOLO-TINY.h5 INTO MODEL FILE
%cd /content/Model
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-
tiny.h5

#LOAD FILES AND STORE THEM IN VARIABLES
model_path = "/content/Model/yolo-tiny.h5"
input_path = "/content/Input/Sheffield_city_centre_x1650-1630x796.jpg"
output_path = "/content/Output/newimg.jpeg"

# FIT THE MODEL
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)

# LOAD THE MODEL
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=input_path,
output_image_path= output_path)

# MODEL TRAINED
# DETERMINE THE PERCENTAGE PROBABILITY FOR EACH OBJECT
for eachObject in detections:
print(eachObject['name'] , ":" , eachObject['percentage_probability'])

# TO DISPLAY THE OUTPUT ON SCREEN
Image(filename = output_path)





