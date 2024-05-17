import cv2
import numpy as np


def detect_green_and_draw_rectangles():
    """
    Detects green color in the webcam feed and draws rectangles around it.

    Returns:
        None
    """
    # Open the webcam
    cap = cv2.VideoCapture("http://192.168.1.33:8081")
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range for the color green in HSV
        lower_green = np.array([70, 70, 70])
        upper_green = np.array([80, 255, 255])

        # Create a mask for the green color
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around detected contours
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "We are not alone", (75, 100), fontFace=1, fontScale=2, color=(0, 255, 0),
                            thickness=2)

        # Display the resulting frame
        cv2.imshow('marvin', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_green_and_draw_rectangles()
