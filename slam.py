import cv2

W = 1920 // 2
H = 1080 // 2

def process_frame():
    # open video
    cap = cv2.VideoCapture("video_examples/test_city_bus.mp4")
   
    # check if video opened successfully
    if not cap.isOpened():
        print("err: Could not open video")
        return
   
    while cap.isOpened():
        # read frame, if no frame, end
        ret, frame = cap.read()
       
        if not ret:
            break
       
        frame = cv2.resize(frame, (W, H))
        cv2.imshow('frame', frame)
       
        # press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frame()