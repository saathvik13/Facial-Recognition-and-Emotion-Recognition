import face_recognition
import os
import cv2
import pickle

TOLERANCE = 0.5
FRAME_THICK = 1

FONT_THICK = 2


KNOWN_DIR = "Known"

known_faces=[]
known_names=[]

print("Started Training")
for name in os.listdir(KNOWN_DIR):
    for f in os.listdir(f"{KNOWN_DIR}/{name}"):
        image=face_recognition.load_image_file(f"{KNOWN_DIR}/{name}/{f}")
        try:
            encoding=face_recognition.face_encodings(image,num_jitters=50)
            print("Name ",name," ",f)
            print("Encoding ",encoding)
            known_faces.append(encoding[0])
            known_names.append(name)

        except:
            pass

print("Stopped Training")

with open('faces.pickle','wb') as f:
    pickle.dump(known_faces,f,protocol=pickle.HIGHEST_PROTOCOL)

with open('names.pickle','wb') as f:
    pickle.dump(known_names,f,protocol=pickle.HIGHEST_PROTOCOL)

print("Saved")
