import numpy as np
import cv2

cap = cv2.VideoCapture(0)


width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))
# out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        print(str(frame.shape[0]) + " " + str(frame.shape[1]))
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()