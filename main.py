import cv2 as cv
import numpy as np 
import time 
import pickle

def detectCascade(xmlModel, frame):
    mlp = pickle.load(open('./data/models/BallDetect.sav', 'rb'))
    scaleFactor =  1.1
    minNeighbors = 4
    minSize = (90,90)
    maxSize = (250, 250)
    alpha = 1.3 # Độ tương phản
    beta = 1.5 # Độ sáng
    detectBall = xmlModel.detectMultiScale(frame,
                                           scaleFactor=scaleFactor,
                                           minNeighbors=minNeighbors,
                                           minSize=minSize, 
                                           maxSize = maxSize,
                                           flags=cv.CASCADE_SCALE_IMAGE)
    if len(detectBall) == 0:
        return 0,0,0,0
        
    for x, y, w, h in detectBall:
        # Crop RoI
        roi = frame[int(y):int(y+h),int(x):int(x+w)]
        # cv.imshow("Cascade", roi)
        roi =  cv.resize(roi, (50, 50))
        roi = cv.convertScaleAbs(roi, alpha=alpha, beta=beta)
        # Normalize
        roi = roi.astype(float)
        cv.normalize(roi, roi, 0, 1.0, cv.NORM_MINMAX)
        # Reshape
        input = np.reshape(roi, (1,2500))
        # Predict
        result =  mlp.predict(input)
        if result == 0:
            return 0, 0, 0, 0
        else:
            print(f"x= {x}; y = {y}; w = {w}; h = {h}")
            return x,y,w,h
        
def main():
    srcXmlModel = "./data/models/cascade.xml"
    xmlModel = cv.CascadeClassifier(srcXmlModel)
    cap = cv.VideoCapture('./data/videos/Video_Ball_1.mp4') # Video quả bóng tâng
    # cap = cv.VideoCapture('./data/videos/Video_Ball_2.mp4') # Video quả bóng chỉ lăn
    # cap = cv.VideoCapture('./data/videos/Video_Ball_3.mp4') # Video quả bóng chỉ lăn và dội ngược lại
    if not cap.isOpened():
        print('Error!')
        exit()
    kltParameters = dict(winSize = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) 
    
    isDetected = False
    fps = 60
    prev = 0
    while True:
        timeElapsed = time.time() - prev
        if timeElapsed > 1./fps:
            prev = time.time()
            ret, frame = cap.read()
            if not ret:
                print("End!")
                break
            if (isDetected == False):
                prevFrame = frame
                prevGray = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)
                x, y, w, h = detectCascade(xmlModel, prevGray)
                if w != 0 and h != 0:
                    roi = prevGray[int(y):int(y+h),int(x):int(x+w)]
                    cv.imshow("Test", roi)
                    keyPoint0 = cv.goodFeaturesToTrack(roi, 50, 0.4, 10, blockSize=7)
                    for point in keyPoint0:
                        point[0,0] = point[0,0] + x
                        point[0,1] = point[0,1] + y
                    isDetected =  True
                mask = np.zeros_like(prevFrame)
            if (isDetected == True):
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                keyPoint1, st, err = cv.calcOpticalFlowPyrLK(prevGray, gray, keyPoint0, None, **kltParameters)
                if err is None:
                    isDetected =False
                    
                if keyPoint1 is not None:
                    newPoint = keyPoint1[st==1]
                    oldPoint = keyPoint0[st==1]
                
                for i, (new, old) in enumerate(zip(newPoint, oldPoint)):
                    xPoint, yPoint = new.ravel()
                    xPoint = int(xPoint)
                    yPoint = int(yPoint)
                    xOldPoint, yOldPoint = old.ravel()
                    xOldPoint = int(xOldPoint)
                    yOldPoint = int(yOldPoint)
                    mask = cv.line(mask, (xPoint, yPoint), (xOldPoint,yOldPoint),(0,0,255), 2)
                    print(f"X = {xPoint}, Y = {yPoint}")
                    frame = cv.circle(frame, (xPoint, yPoint), 5, (255,0,0), 2)
                frame = cv.add(frame,mask)
                prevGray = gray.copy()
                keyPoint0 = newPoint.reshape(-1, 1, 2)
            # cv.imshow('mask', mask)
            cv.imshow('video', frame)
            if cv.waitKey(1) == ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
    