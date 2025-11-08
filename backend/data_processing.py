'''
process livefeed data from webcam 

1) preprocess data to eliminate noise 
2) face recognitiion 
    i) if face detected within radius -> run similarity check against existing profiles, 
        run live audio processing, send to LLM for summarization, store
        summarization and time in DB
3) Live video feed processing
    i) capture video feed in frames only when significant scene change
    -> send to LLM for summarization
        -> store summarization and time in DB
    ii) have temporary current relevant conversation recording (for the individual to 
    query their current conversation, indicate redundancy )
4) Data handling and storage
    i) thorughout the day scenes and events summarized
    ii) temporary current relevant conversation recording 
    iii) summarized audio data associated with relevant profile
    iv) facial profiles 
'''
import cv2


def webcam_processing():
#change the indice to the webcam index
cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("BRIO feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    webcam_processing()

if __name__ == "__main__":
    main()