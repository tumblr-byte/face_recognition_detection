import cv2
import os
import face_recognition

# Paths and class names
images_folder_path = "Images"
class_names = ["Bill Gates", "Emma Watson", "Scarlett Johansson"]

known_faces = []
known_face_names = []

# Load known faces
for class_name in class_names:
    class_folder_path = os.path.join(images_folder_path, class_name)

    if os.path.isdir(class_folder_path):
        for image_path in os.listdir(class_folder_path):
            image_name = os.path.join(class_folder_path, image_path)

            if os.path.isfile(image_name):
                image = face_recognition.load_image_file(image_name)
                encodings = face_recognition.face_encodings(image)

                if encodings:  # Ensure there's at least one face
                    known_faces.append(encodings[0])
                    known_face_names.append(class_name)

# Load unknown face
unknown_face = face_recognition.load_image_file("img1.jpg")
unknown_face_encodings = face_recognition.face_encodings(unknown_face)
face_locations = face_recognition.face_locations(unknown_face)

# Verify if a face was detected
if unknown_face_encodings and face_locations:
    unknown_face_encoding = unknown_face_encodings[0]

    # Compare with known faces
    results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
    face_distances = face_recognition.face_distance(known_faces, unknown_face_encoding)

    # Identify the best match
    best_match_index = None
    if True in results:
        best_match_index = results.index(True)
        matched_name = known_face_names[best_match_index]
        match_distance = face_distances[best_match_index]

        print(f"Match found: {matched_name} with a distance of {match_distance:.2f}")
    else:
        print("No match found.")

    # Draw rectangles around faces
    for (top, right, bottom, left) in face_locations:
        # Convert to OpenCV format (BGR)
        image_with_faces = cv2.cvtColor(unknown_face, cv2.COLOR_RGB2BGR)

        # Draw rectangle
        cv2.rectangle(image_with_faces, (left, top), (right, bottom), (0, 255, 0), 2)

        # Add label
        label = matched_name if best_match_index is not None else "Unknown"
        cv2.putText(
            image_with_faces,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    # Save the output image
    output_path = "output.jpg"
    cv2.imwrite(output_path, image_with_faces)
    print(f"Result saved to {output_path}")
else:
    print("No face detected in the unknown image.")





    






