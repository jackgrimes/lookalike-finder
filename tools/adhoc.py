import face_recognition
import os

if __name__ == "__main__":

    image1 = face_recognition.load_image_file(os.path.join(os.getcwd(), os.path.join(os.path.join('inputs', 'image1.jpg'))))
    encodings1 = face_recognition.face_encodings(image1)

    image2 = face_recognition.load_image_file(os.path.join(os.getcwd(), os.path.join(os.path.join('inputs', 'image2.jpg'))))
    encodings2 = face_recognition.face_encodings(image2)

    image3 = face_recognition.load_image_file(os.path.join(os.getcwd(), os.path.join(os.path.join('inputs', 'image3.jpg'))))
    encodings3 = face_recognition.face_encodings(image3)
    
    print("Face distance between 1 and 2: " + str(face_recognition.face_distance(encodings1, encodings2[0])[0]))
    print("Face distance between 1 and 3: " + str(face_recognition.face_distance(encodings1, encodings3[0])[0]))
    print("Face distance between 2 and 3: " + str(face_recognition.face_distance(encodings2, encodings3[0])[0]))
