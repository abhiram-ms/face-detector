import cv2
from random import randrange

#trained face data
trained_front_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
trained_smile_data = cv2.CascadeClassifier("haarcascade_smile.xml")
trained_upper_body_data = cv2.CascadeClassifier("haarcascade_upperbody.xml")


#img = cv2.imread("images6.jfif")
webcam = cv2.VideoCapture(0)


while webcam.isOpened():
    success_frame_read, frame = webcam.read()

    #if error occurs break
    if not success_frame_read:
        break

    #greyscale the image
    grey_scale_img_frontface = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey_scale_upperbody = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect faces
    front_face_coordinates = trained_front_face_data.detectMultiScale(grey_scale_img_frontface)
    upper_body_coordinates = trained_upper_body_data.detectMultiScale(grey_scale_upperbody)

    print("upper :",upper_body_coordinates)

    #draw rectangle
    for(x,y,w,h) in front_face_coordinates:
        #draw rectangle
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,256,0),4)
        #detecting all the parts of body
        the_face = frame[y:y+h, x:x+w]
        grey_smile_face = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
        #detect smiles
        smile_coordinates = trained_smile_data.detectMultiScale(grey_smile_face,scaleFactor=1.3,minNeighbors=3)
        for(x_,y_,w_,h_) in smile_coordinates:
            cv2.rectangle(the_face,(x_,y_), (x_+w_, y_+h_), (0,0,256),2)
            cv2.putText(frame,"smiling",(x,y+h+40),fontScale=3,
                fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,0,0))
            
    for(x,y,w,h) in upper_body_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,0),4)

    cv2.imshow("detection App (press q to exit)", frame)
    key =cv2.waitKey(1)

    #key q pressed quit
    if(key == 81 or key == 113):
        break

#release the webcam
webcam.release()
cv2.destroyAllWindows()

print("completed in success")
