import cv2
import time
import imutils
cam=cv2.VideoCapture(0)
time.sleep(1)

firstframe=None
area=500
val=0
#process one capture the image 
while True:
    _,img=cam.read()
    #displaying the text inside of the camera feed
    text="Normal"
    img=imutils.resize(img,width=300,height=300)
    grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianimg=cv2.GaussianBlur(grayimage,(21,21),0)

    if firstframe is None:
        firstframe=gaussianimg
        continue
    imgdiff=cv2.absdiff(firstframe,gaussianimg)
    thres=cv2.threshold(imgdiff,25,255,cv2.THRESH_BINARY)[1]
    thres=cv2.dilate(thres,None,iterations=2)

    #cnts=cv2.findContours(thres.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts,hierarchy= cv2.findContours(thres.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        
        if cv2.contourArea(c) < area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        val+=1
        text="Moving object has detected"+str(val)
        print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    
    cv2.imshow("Camera ON",img) #camera on position
    cv2.imshow("camera  ON in grayimage",grayimage)
    cv2.imshow("camera ON in gaussianimage",gaussianimg)
    key=cv2.waitKey(1)&0xFF
    if key==ord("q"):
        print("thanks for quiting the projectt");
        break
cam.release()
#destroy all the windows foe closing the whole project when your clicking the x
cv2.destroyAllWindows()
