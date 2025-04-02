import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose Estimation - Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Estimation - Video", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_webcam():
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Estimation - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Process an Image")
    print("2. Process a Video")
    print("3. Use Webcam")
    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        image_path = input("Enter image path: ")
        process_image(image_path)
    elif choice == "2":
        video_path = input("Enter video path: ")
        process_video(video_path)
    elif choice == "3":
        process_webcam()
    else:
        print("Invalid choice!")
