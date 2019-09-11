import os
import face_recognition
from operator import itemgetter
import sys
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import datetime

from configs import lfw_path, data_path, ALLOWED_PICTURE_FILE_TYPES
from utils import timesince, get_user_inputs

def run():

    i_max, max_len_closest_matches, image_name, person_count = get_user_inputs()

    try:
        image_name
    except NameError:
        print("No file selected for comparisons!")
    else:

        file_count = 0
        person_no = 0
        for _, dirs, files in os.walk(lfw_path):
            if person_no <= i_max:
                person_no += 1
                file_count += len([x for x in files if x != "Thumbs.db"])
            else:
                break

        if i_max < person_count:
            person_count = i_max
            file_count = sum([len(files) for r, d, files in os.walk(lfw_path)][1:(i_max + 1)])
            print(str(file_count) + " images of " + str(person_count) + " people to compare with " + image_name)

        runstr = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M") + '_run_top_' + str(max_len_closest_matches) + \
                 "_lookalikes_for_" + image_name.replace(".", "_") + "_from_" + str(file_count) + "_images_of_" + str(
            person_count) + "_people"

        image = face_recognition.load_image_file(os.path.join(os.path.join(data_path, 'inputs'), image_name))
        encodings = face_recognition.face_encodings(image)
        closest_matches = []

        start_time = datetime.datetime.now()

        i = 1
        j = 1
        for folder in os.listdir(lfw_path):
            for file in os.listdir(os.path.join(lfw_path, folder)):
                print("Comparing image " + str(j) + " of " + str(file_count) + " (" + str(
                    round(100 * (j - 1) / file_count, 2)) +
                      "% complete), person " + str(i) + " of " + str(person_count) + ", file " + os.path.join(
                    os.path.join(folder, file)) + " with " + image_name)
                celeb_image = face_recognition.load_image_file(
                    os.path.join(os.getcwd(), os.path.join(os.path.join(lfw_path, os.path.join(folder, file)))))
                celeb_encodings = face_recognition.face_encodings(celeb_image)
                if len(celeb_encodings) == 0:
                    print("no face found")
                else:
                    filename = os.path.join(folder, file)
                    distance = face_recognition.face_distance(celeb_encodings, encodings[0])[0]
                    print("Distance is: " + str(round(distance, 3)))
                    if len(closest_matches) < max_len_closest_matches:
                        closest_matches.append([filename, distance])
                    else:
                        closest_matches_distances = [x[1] for x in closest_matches]
                        if distance < max(closest_matches_distances):
                            closest_matches.append([filename, round(distance, 3)])
                            closest_matches_distances = [x[1] for x in closest_matches]
                            index_max = max(range(len(closest_matches_distances)),
                                            key=closest_matches_distances.__getitem__)
                            del (closest_matches[index_max])
                    closest_matches_distances = [x[1] for x in closest_matches]
                    index_min = min(range(len(closest_matches_distances)), key=closest_matches_distances.__getitem__)
                    print("Current closest lookalike to " + image_name + " is " + closest_matches[index_min][
                        0] + " with a distance of " + str(round(closest_matches[index_min][1], 3)))
                    print(timesince(start_time, 100 * (j - 1) / file_count) + "\n")
                j += 1
            i += 1
            if i > i_max:
                break

        closest_matches_sorted = sorted(closest_matches, key=itemgetter(1))

        print("\n#\nRESULTS\n#\n")
        print(closest_matches_sorted)

        df = pd.DataFrame(closest_matches_sorted, columns=['person', 'facial distance'])
        df.to_csv(os.path.join(os.path.join(data_path, 'results'), runstr + '.csv'))

        image_paths = [os.path.join(lfw_path, x[0]) for x in closest_matches_sorted]

        original = cv2.imread(os.path.join(os.path.join(data_path, 'inputs'), image_name))
        original = cv2.resize(original, (250, 250))

        images = [original] + [cv2.imread(x) for x in image_paths]
        combined = np.concatenate(images, axis=1)
        cv2.imwrite(os.path.join(os.path.join(data_path, 'results'), runstr + '.jpg'), combined)


if __name__ == "__main__":
    run()
