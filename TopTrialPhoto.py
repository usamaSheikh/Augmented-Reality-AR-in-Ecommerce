import cv2
import imutils
import numpy as np
import tkinter as tk
from tkinter import filedialog
import CollectionOfTops as cc
import random

# Global variables for drag-and-drop and resizing
dragging = False
start_x, start_y = 0, 0
cloth_x, cloth_y = 200, 200
cloth_width, cloth_height = 300, 300

# Load face detector for automatic fitting
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def mouse_callback(event, x, y, flags, param):
    global dragging, start_x, start_y, cloth_x, cloth_y, cloth_width, cloth_height

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start dragging
        if cloth_x < x < cloth_x + cloth_width and cloth_y < y < cloth_y + cloth_height:
            dragging = True
            start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Update clothing position during drag
        dx, dy = x - start_x, y - start_y
        cloth_x += dx
        cloth_y += dy
        start_x, start_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging
        dragging = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            # Zoom out (Shift + right-click)
            if cloth_width > 50 and cloth_height > 50:  # Prevent shrinking below a minimum size
                cloth_width -= 10
                cloth_height -= 10
        else:
            # Zoom in (right-click)
            cloth_width += 10
            cloth_height += 10


def fit_prop_to_body(frame):
    """
    Automatically adjust the size and position of the prop (clothing item)
    to fit the body based on face detection and estimated torso dimensions.
    """
    global cloth_x, cloth_y, cloth_width, cloth_height

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

    if len(faces) > 0:
        # Assume the first detected face is the main subject
        (x, y, w, h) = faces[0]
        print(f"Detected face at X:{x}, Y:{y}, W:{w}, H:{h}")

        # Estimate shoulder width and torso dimensions based on face size
        shoulder_width = int(w * 2)  # Use face width to estimate shoulder width
        torso_height = int(h * 3)   # Use face height to estimate torso length

        # Adjust the shirt (prop) size
        cloth_width = shoulder_width
        cloth_height = torso_height

        # Position the shirt just below the face
        cloth_x = x - (shoulder_width // 4)  # Center the shirt horizontally
        cloth_y = y + h                      # Place the shirt vertically below the face
    else:
        print("No face detected. Ensure the face is clear and the image is well-lit.")


def select_new_image():
    """
    Open a file dialog to select a new image and return the loaded image.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if file_path:
        return cv2.imread(file_path)
    else:
        return None


def fashion():
    global cloth_x, cloth_y, cloth_width, cloth_height

    # Load tops
    images = cc.loadTops()
    currentClothId = 0

    # Ask user to select an image initially
    original_frame = select_new_image()
    if original_frame is None:
        print("No image selected. Exiting.")
        return

    # Resize the image for better visibility
    frame_to_try = imutils.resize(original_frame.copy(), width=800)

    # Set up OpenCV mouse callback
    cv2.namedWindow("Try Tops on Your Photo")
    cv2.setMouseCallback("Try Tops on Your Photo", mouse_callback)

    while True:
        # Create a fresh copy of the original image to overlay the top
        frame_to_try = imutils.resize(original_frame.copy(), width=800)

        # Get the current top and resize it
        top = images[currentClothId]
        top = cv2.resize(top, (cloth_width, cloth_height))

        # Ensure the overlay fits within the image dimensions
        f_height, f_width, _ = frame_to_try.shape
        if cloth_y + cloth_height > f_height:
            cloth_height = f_height - cloth_y
        if cloth_x + cloth_width > f_width:
            cloth_width = f_width - cloth_x

        # Apply threshold and masks
        topGray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(topGray, 200, 255, cv2.THRESH_BINARY_INV)
        inverted_mask = cv2.bitwise_not(mask)
        roi = frame_to_try[cloth_y:cloth_y + cloth_height, cloth_x:cloth_x + cloth_width]
        image_background = cv2.bitwise_and(roi, roi, mask=inverted_mask)
        image_foreground = cv2.bitwise_and(top, top, mask=mask)

        # Combine background and foreground
        overlay = cv2.add(image_background, image_foreground)

        # Place the combined result back onto the original image
        frame_to_try[cloth_y:cloth_y + cloth_height, cloth_x:cloth_x + cloth_width] = overlay

        # Add instructions text
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_text, y_text = 10, 20
        cv2.putText(frame_to_try, "Press 'n' for next top, 'p' for previous, 'Enter' to fit, 's' to save, 'c' to change image, 'Esc' to exit.",
                    (x_text, y_text), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_to_try, "Left-click to drag, right-click to zoom in, Shift+right-click to zoom out.",
                    (x_text, y_text + 20), font, 0.5, (255, 255, 255), 1)

        # Display the image
        cv2.imshow("Try Tops on Your Photo", frame_to_try)

        # Handle key inputs
        key = cv2.waitKey(10)
        if key & 0xFF == ord('n'):  # Next top
            currentClothId = (currentClothId + 1) % len(images)
        if key & 0xFF == ord('p'):  # Previous top
            currentClothId = (currentClothId - 1) % len(images)
        if key & 0xFF == ord('s'):  # Save on pressing 's'
            rand = random.randint(1, 999999)
            cv2.imwrite('output/' + str(rand) + '.png', frame_to_try)
            print(f"Image saved as 'output/{rand}.png'")
        if key == 13:  # Fit the prop when Enter is pressed
            fit_prop_to_body(original_frame)
        if key & 0xFF == ord('c'):  # Change image on pressing 'c'
            new_image = select_new_image()
            if new_image is not None:
                original_frame = new_image
                cloth_x, cloth_y = 200, 200  # Reset position of the clothing

        if key == 27:  # Exit on pressing 'Esc'
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()


fashion()
