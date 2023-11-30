import pathlib
import sys
import time

from draw_utils import *
from facemesh import *


ENABLE_EDGETPU = False

MODEL_PATH = pathlib.Path("./models/")
if ENABLE_EDGETPU:

    MESH_MODEL = "cocompile/face_landmark_192_full_integer_quant_edgetpu.tflite"
else:
    
    MESH_MODEL = "face_landmark.tflite"




# turn on camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
ret, init_image = cap.read()
if not ret:
    sys.exit(-1)

# instantiate facemesh models

face_mesher = FaceMesher(model_path=str((MODEL_PATH / MESH_MODEL)), edgetpu=ENABLE_EDGETPU)







# endless loop
while True:
    start = time.time()
    ret, face = cap.read()
    face_input = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    
    landmarks, scores = face_mesher.inference(face_input)
    face_output = draw_mesh(face_input, landmarks, contour=True)

    # put fps
    
    result = cv2.cvtColor(face_output, cv2.COLOR_RGB2BGR)

    cv2.imshow('demo', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
