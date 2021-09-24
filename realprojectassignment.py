import cv2
import time
import imutils
#camera
cam=cv2.VideoCapture(0)
time.sleep(1)

firstframe=None
area=500
v=0

while True:
    _,img=cam.read()
    text="normal" #for empty scrren its shows normal on the camera feed

    img=imutils.resize(img,width=400)
    grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianimage=cv2.GaussianBlur(grayimage,(21,21),0)

    if firstframe is None:
        firstframe=gaussianimage
        continue

    imgdiff=cv2.absdiff(firstframe,gaussianimage)
    thres=cv2.threshold(imgdiff,25,255,cv2.THRESH_BINARY)[1]
    thres=cv2.dilate(thres,None,iterations=2)

    cnts=cv2.findContours(thres.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    cnts=imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c)<area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        v+=1
        text="Moving object detected"+" "+str(v)
        print(text)
        cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,25),2)

        cv2.imshow("camera On.jpg",img)
        key=cv2.waitKey(1)&0xFF
        if key==ord("q"):
            print("thanks for using my project")
            break

cam.release()


cv2.destroyAllWindows()

        




        
        
    
