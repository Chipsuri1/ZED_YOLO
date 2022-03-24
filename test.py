import cv2

cap= cv2.VideoCapture(0)

# def make_720p():
#     cap.set(3, 1280)
#     cap.set(4, 720)
#
# make_720p()
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc('M','P','E','G')

# out = cv2.VideoWriter('./processed.avi', codec, 20.0, (720,1280))
writer= cv2.VideoWriter('basicvideo.mp4', codec, 20, (width,height))


while True:
    ret,frame= cap.read()

    writer.write(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()