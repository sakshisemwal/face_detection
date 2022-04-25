import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontface_default.xml')
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('FACE DETECTOR', img)
   
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()

#importing cv2 that is open-cv library- which aims at real time computer vision.
#haar cascade is the ML approach, positive and negative pictures are fed to train the classifier.
#load the classifier ( in face_cascade )

#cv2 processes image in BGR not in RGB, convert the image to gray scale 
