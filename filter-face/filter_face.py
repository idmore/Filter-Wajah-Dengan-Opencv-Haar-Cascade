import cv2 

face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt.xml')

#source = https://www.pexels.com/photo/people-girl-design-happy-35188/
frame = cv2.imread('images/baby.png')

#source = https://www.digitalocean.com/community/tutorials/how-to-apply-computer-vision-to-build-an-emotion-based-dog-filter-in-python-3 
face_mask = cv2.imread('images/dog.png') 

face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3) 
for (x,y,w,h) in face_rects:
    h, w = int(1.5*h), int(2.0*w) 
    y -= int(0.1*h+22)
    x = int(x-33)

    frame_roi = frame[y:y+h, x:x+w]
    face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA) 
    gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY) 
    _, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY_INV) 
    mask_inv = cv2.bitwise_not(mask) 
    masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask) 
    masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv) 
    
    frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)
 
cv2.imwrite('images/output.png', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
