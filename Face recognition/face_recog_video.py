import face_recognition
import os
import cv2
import pickle

TOLERANCE = 0.5
FRAME_THICK = 1

FONT_THICK = 2


    
with open('faces.pickle', 'rb') as handle:
    known_faces = pickle.load(handle)

with open('names.pickle', 'rb') as handle:
    known_names = pickle.load(handle)


video = cv2.VideoCapture(0)
model = "cnn"

b=True
while b:

    ret,image = video.read()
    locations = face_recognition.face_locations(image,model=model)
    encoding=face_recognition.face_encodings(image,locations)
    
##    cv2.imshow("image",image)
    
##    print(locations)

    for encod,loc in zip(encoding,locations):
        results = face_recognition.compare_faces(known_faces,encod,TOLERANCE)
        print(results)
        match = None
        if True in results:
            match = known_names[results.index(True)]

            top_left = (loc[3],loc[0])
            bottom_right = (loc[1],loc[2])
            color=[0,255,0]

            cv2.rectangle(image, top_left,bottom_right,color,FRAME_THICK)

            top_left = (loc[3]+10,loc[2]+10)
            bottom_right = (loc[1]+15,loc[2]+30)
            color=[0,255,0]

            cv2.rectangle(image, top_left,bottom_right,color)
            cv2.putText(image,match,(loc[3]+15,loc[2]+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),FONT_THICK)
            
        cv2.imshow("images",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            b=False
            break
##        cv2.waitKey(0)
            

            
        
    
