import dlib
import cv2

# path of pre-trained model 
predictor_path = './shape_predictor_68_face_landmarks.dat'

# detects faces
detector = dlib.get_frontal_face_detector()
# predicts facial landmarks
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detecting all faces
    rects = detector(gray, 0)
    for rect in rects:
        # detecting facial landmarks
        shape = predictor(gray, rect)
        # drawing the predicted points 
        for i in range(68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0, 0, 255), -1)
        # drawing jawline
        for i in range(16):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y),(0,0,255),1)
        # drawing eyebrow 1
        for i in range(17,21):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y),(0,0,255),1)
        # drawing eyebrow 2
        for i in range(22,26):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y),(0,0,255),1)
            
    cv2.imshow("image", frame)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
