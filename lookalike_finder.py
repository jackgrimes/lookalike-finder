import os
import face_recognition
from operator import itemgetter
import sys
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import datetime

def timesince(time, percent_done):
    diff = datetime.datetime.now() - time
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    try:
        remaining_diff =  (diff/percent_done)*100 - diff
        remaining_days, remaining_seconds = remaining_diff.days, remaining_diff.seconds
        remaining_hours = remaining_days * 24 + remaining_seconds // 3600
        remaining_minutes = (remaining_seconds % 3600) // 60
        remaining_seconds = remaining_seconds % 60
        return (str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds taken so far" + "\n" +
                "Estimated " + str(remaining_hours) + " hours, " + str(remaining_minutes) + " minutes, " + str(remaining_seconds) + " seconds to completion")
    except:
        return ("Cannot calculate times done and remaining at this time")

def run():

    max_len_closest_matches = input("How many closest matches do you want to see? (leave blank for 10)")
    i_max = input("How many people's faces do you want to compare with? (leave blank for all)")

    if max_len_closest_matches == "":
        max_len_closest_matches = 10

    max_len_closest_matches = int(max_len_closest_matches)

    person_count = len(([len(files) for r, d, files in os.walk(r"C:\dev\data\lfw")])) - 1

    if i_max == "":
        i_max = person_count

    i_max = int(i_max)

    potential_images = [x for x in os.listdir(r"C:\dev\data\lookalike_finder\inputs") if 'jpg' in x]

    for x in potential_images:
        response = input("Comparing " + x + "? (y for yes, n or no)")
        while response not in ['y', 'n']:
            print("enter y or n")
            response = input("Comparing " + x + "? (y for yes, n or no)")
        if response == "y":
            image_name = x
            break

    try:
        image_name
    except NameError:
        print("No file selected for comparisons!")
    else:

        file_count = 0
        person_no = 0
        for _, dirs, files in os.walk(r'C:\dev\data\lfw'):
            if person_no <= i_max:
                person_no += 1
                file_count += len([x for x in files if x != "Thumbs.db"])
            else:
                break

        if i_max < person_count:
            person_count = i_max
            file_count = sum([len(files) for r, d, files in os.walk(r"C:\dev\data\lfw")][1:(i_max+1)])
            print(str(file_count) + " images of "+ str(person_count) + " people to compare with " + image_name)

        runstr = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M") + '_run_top_' + str(max_len_closest_matches) + \
                 "_lookalikes_for_"+ image_name.replace(".", "_") + "_from_" + str(file_count) + "_images_of_"+ str(person_count) + "_people"

        image = face_recognition.load_image_file(os.path.join(r'C:\dev\data\lookalike_finder\inputs', image_name))
        encodings = face_recognition.face_encodings(image)
        closest_matches = []

        start_time = datetime.datetime.now()

        i = 1
        j = 1
        for folder in os.listdir(r'C:\dev\data\lfw'):
            for file in os.listdir(os.path.join(r'C:\dev\data\lfw', folder)):
                print("Comparing image " + str(j) + " of " + str(file_count) + " (" + str(round(100*(j-1)/file_count, 2)) +
                      "% complete), person "+ str(i) + " of " + str(person_count) + ", file " + os.path.join(os.path.join(folder, file)) + " with " + image_name)
                celeb_image = face_recognition.load_image_file(os.path.join(os.getcwd(), os.path.join(os.path.join(r'C:\dev\data\lfw', os.path.join(folder, file)))))
                celeb_encodings = face_recognition.face_encodings(celeb_image)
                if len(celeb_encodings) == 0:
                    print("no face found")
                else:
                    filename = os.path.join(folder, file)
                    distance = face_recognition.face_distance(celeb_encodings, encodings[0])[0]
                    print("Distance is: " + str(round(distance,3)))
                    if len(closest_matches) < max_len_closest_matches:
                        closest_matches.append([filename, distance])
                    else:
                        closest_matches_distances = [x[1] for x in closest_matches]
                        if distance < max(closest_matches_distances):
                            closest_matches.append([filename, round(distance, 3)])
                            closest_matches_distances = [x[1] for x in closest_matches]
                            index_max = max(range(len(closest_matches_distances)), key=closest_matches_distances.__getitem__)
                            del(closest_matches[index_max])
                    closest_matches_distances = [x[1] for x in closest_matches]
                    index_min = min(range(len(closest_matches_distances)), key=closest_matches_distances.__getitem__)
                    print("Current closest lookalike to " + image_name + " is " + closest_matches[index_min][0] + " with a distance of " + str(round(closest_matches[index_min][1],3)))
                    print(timesince(start_time, 100*(j-1)/file_count)+"\n")
                j += 1
            i += 1
            if i > i_max:
                break

        closest_matches_sorted = sorted(closest_matches, key=itemgetter(1))

        print("\n#\nRESULTS\n#\n")
        print(closest_matches_sorted)

        df = pd.DataFrame(closest_matches_sorted, columns=['person', 'facial distance'])
        df.to_csv(os.path.join(r'C:\dev\data\lookalike_finder\results', runstr + '.csv'))

        image_paths = [os.path.join(os.getcwd(), os.path.join(r"C:\dev\data\lfw", x[0])) for x in closest_matches_sorted]

        original = cv2.imread(os.path.join(r'C:\dev\data\lookalike_finder\inputs', image_name))
        original = cv2.resize(original, (250, 250))

        images = [original] + [cv2.imread(x) for x in image_paths]
        combined = np.concatenate(images, axis=1)
        cv2.imwrite(os.path.join(r'C:\dev\data\lookalike_finder\results', runstr+'.jpg'), combined)


if __name__ == "__main__":
    run()