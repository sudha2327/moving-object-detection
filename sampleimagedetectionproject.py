import cv2
import time
import imutils

cam = cv2.VideoCapture(0)
time.sleep(1)
firstFrame = None
area = 300
a = 0

while True:
    _, img = cam.read()
    text = "Normal"
    img = imutils.resize(img, width=400)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianimg = cv2.GaussianBlur(grayimg, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gaussianimg
        continue
    imgDiff = cv2.absdiff(firstFrame, gaussianimg)
    thresholdimg = cv2.threshold(imgDiff, 100, 255, cv2.THRESH_BINARY)[1]
    thresholdimg = cv2.dilate(thresholdimg, None, iterations=2)
    cnts = cv2.findContours(thresholdimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        a += 1
        text = "Moving object detected" + " " + str(a)
    print(a)
    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("camerafeed.jpg", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
