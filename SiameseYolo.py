import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pyautogui as pg
import keyboard



model = load_model('./weights-improvement-91-0.93.hdf5')
referenceGesture=None
referenceGestureRecordFlag=False
snapShot=False

#Getting the camera
cap = cv2.VideoCapture(0)

#making the yolo nn architecture
net = cv2.dnn.readNet("./yolov4-tiny-hand_best.weights","./yolov4-tiny-testing.cfg")
#Getting reference to the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#Making an array to contain the name of objects in final layer
classes = [""]


while(True):
    #Getting image from the camera
    ret, img = cap.read()
    #img = cv2.flip(img, 1)
    img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
    height, width, channels = img.shape
    

    if keyboard.is_pressed("r"):
        referenceGestureRecordFlag=not referenceGestureRecordFlag
    elif keyboard.is_pressed('s'):
        snapShot=True


    ##########################YOLO CODE STARTS FROM HERE###########################
    #Processing the image (i.e making the blob)
    #dividing by 255, reshaping into 288 by 288
    blob = cv2.dnn.blobFromImage(img, 0.00392 , (288, 288), (0, 0, 0), True, crop=False)
    
    #Passing the blob to the neural network
    net.setInput(blob)
    #Collecting the feature maps from the output layers
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        
        for detection in out:
            #The three lines below is used for finding confidence value
            scores=detection[5:] 
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence>0.3:
                #It means we are considering a detected object                    
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                                        
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # The boxes array contain the detected boxes
        
        # We further get the indexes of boxes after Non Max Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        
        font = cv2.FONT_HERSHEY_PLAIN
        
        #Now drawing the boxes from the images
        
        for i in range(len(boxes)):
            if i in indexes:
                try:
                    x, y, w, h = boxes[i]
                    #print("x1,y1 is= "+str(x)+", "+str(y))
                    label = str(classes[class_ids[i]])
                    #print("Label ",label)
                    color = (244,0,0)
                    
                    
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
                    croppedImg = img[y:y+h, x:x+w]
                    croppedImg=cv2.resize(croppedImg,(100,100))
                    croppedImg=cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY )
                    cv2.imshow("CroppedImage",croppedImg)

                    #Processing it for tensorflow model
                    
                    croppedImg = np.stack((croppedImg,)*3, axis=-1)
                    croppedImg=croppedImg.reshape((1,100,100,3))
                    croppedImg=croppedImg/255
                    if(referenceGestureRecordFlag):
                        cv2.putText(img, "Can take snap", (50, 100), font, 3, color, 2)                        
                        referenceGesture=None
                        if(snapShot==True):
                            referenceGesture=croppedImg
                            referenceGestureRecordFlag=False
                            snapShot=False
                    else:
                        
                        if(referenceGesture is  None):
                            pass
                        else:                            
                            y=model.predict([referenceGesture,croppedImg])
                            cv2.putText(img, "Can compare:", (10, 50), font, 3, color, 2)
                            cv2.putText(img, str(y), (10, 80), font, 3, color, 2)
                            print(y)

                                 
                	

                except:
                	pass

        cv2.imshow("Image", img)
    # Display the resulting frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()