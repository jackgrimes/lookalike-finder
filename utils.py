import datetime
import itertools
import os
from operator import itemgetter

import cv2
import face_recognition
import numpy as np
import pandas as pd

from configs import lfw_path, data_path, ALLOWED_PICTURE_FILE_TYPES


def timesince(time, percent_done):
    """

    :param time:
    :param percent_done:
    :return:
    """
    diff = datetime.datetime.now() - time
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    try:
        remaining_diff = (diff / percent_done) * 100 - diff
        remaining_days, remaining_seconds = remaining_diff.days, remaining_diff.seconds
        remaining_hours = remaining_days * 24 + remaining_seconds // 3600
        remaining_minutes = (remaining_seconds % 3600) // 60
        remaining_seconds = remaining_seconds % 60
        return (str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds taken so far" + "\n" +
                "Estimated " + str(remaining_hours) + " hours, " + str(remaining_minutes) + " minutes, " + str(
                    remaining_seconds) + " seconds to completion")
    except:
        return ("Cannot calculate times done and remaining at this time")


def get_user_inputs():
    """

    :return:
    """
    max_len_closest_matches = input("How many closest matches do you want to see? (leave blank for 10)")
    i_max = input("How many people's faces do you want to compare with? (leave blank for all)")

    if max_len_closest_matches == "":
        max_len_closest_matches = 10

    max_len_closest_matches = int(max_len_closest_matches)

    person_count = len(([len(files) for r, d, files in os.walk(lfw_path)])) - 1

    if i_max == "":
        i_max = person_count

    i_max = int(i_max)

    potential_images = [x for x in os.listdir(os.path.join(data_path, 'inputs')) if
                        x.split('.')[1] in ALLOWED_PICTURE_FILE_TYPES]

    for x in itertools.cycle(potential_images):
        response = input("Comparing " + x + "? (y for yes, n or no)")
        while response not in ['y', 'n']:
            print("enter y or n")
            response = input("Comparing " + x + "? (y for yes, n or no)")
        if response == "y":
            image_name = x
            break

    image_path = os.path.join(os.path.join(data_path, 'inputs'), image_name)

    return i_max, max_len_closest_matches, image_name, person_count, image_path


def get_number_of_pics_to_compare_with(i_max, lfw_path, person_count, image_name):
    """

    :param i_max:
    :param lfw_path:
    :return:
    """

    file_count = 0
    person_no = 0

    for root, dirs, files in os.walk(lfw_path):
        if person_no <= i_max:
            person_no += 1
            file_count += len([x for x in files if x != "Thumbs.db"])
        else:
            break

    if i_max < person_count:
        person_count = i_max
        file_count = sum([len(files) for r, d, files in os.walk(lfw_path)][1:(i_max + 1)])
        print(
            "\n" + str(file_count) + " images of " + str(person_count) + " people to compare with " + image_name + "\n")

    return file_count


def get_image_encodings(image_path):
    """

    :param image_name:
    :return:
    """

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    return encodings


def compare_face_with_original(folder, file, encodings, file_count, person_count, image_name, j, i):
    """

    :param folder:
    :param file:
    :param encodings:
    :param file_count:
    :param person_count:
    :param image_name:
    :param j:
    :param i:
    :return:
    """
    print("Comparing image " + str(j) + " of " + str(file_count) + " (" + str(
        round(100 * (j - 1) / file_count, 2)) +
          "% complete), person " + str(i) + " of " + str(person_count) + ", file " + os.path.join(
        os.path.join(folder, file)) + " with " + image_name)

    lfw_face_encodings = get_image_encodings(
        os.path.join(os.getcwd(), os.path.join(os.path.join(lfw_path, os.path.join(folder, file)))))

    if len(lfw_face_encodings) == 0:
        distance = np.NaN
        print("no face found")

    else:
        distance = face_recognition.face_distance(lfw_face_encodings, encodings[0])[0]
        print("Distance is: " + str(round(distance, 3)))

    filename = os.path.join(folder, file)

    return filename, distance


def get_closest_matches_distances(closest_matches):
    """

    :param closest_matches:
    :return:
    """
    closest_matches_distances = [x[1] for x in closest_matches]
    return closest_matches_distances


def add_face_to_closest_matches(filename, distance, closest_matches):
    """

    :param filename:
    :param distance:
    :param closest_matches:
    :return:
    """
    closest_matches.append([filename, distance])
    closest_matches_distances = get_closest_matches_distances(closest_matches)
    index_min = min(range(len(closest_matches_distances)), key=closest_matches_distances.__getitem__)
    return closest_matches, closest_matches_distances, index_min


def add_face_to_closest_matches_if_closer_than_current_closest_mathes(filename, distance, closest_matches,
                                                                      closest_matches_distances, index_min):
    """

    :param filename:
    :param distance:
    :param closest_matches:
    :param closest_matches_distances:
    :return:
    """
    if distance < max(closest_matches_distances):
        closest_matches.append([filename, round(distance, 3)])
        index_max = max(range(len(closest_matches_distances)),
                        key=closest_matches_distances.__getitem__)
        del (closest_matches[index_max])
        closest_matches_distances = get_closest_matches_distances(closest_matches)
        index_min = min(range(len(closest_matches_distances)), key=closest_matches_distances.__getitem__)

    return closest_matches, closest_matches_distances, index_min


def print_update(image_name, closest_matches, index_min, start_time, file_count, j):
    """

    :param image_name:
    :param closest_matches:
    :param index_min:
    :param start_time:
    :param file_count:
    :return:
    """
    print("Current closest lookalike to " + image_name + " is " + closest_matches[index_min][
        0] + " with a distance of " + str(round(closest_matches[index_min][1], 3)))
    print(timesince(start_time, 100 * (j - 1) / file_count) + "\n")


def compare_with_all_other_images(lfw_path, file_count, person_count, encodings, image_name, max_len_closest_matches,
                                  start_time, i_max):
    """

    :param lfw_path:
    :param file_count:
    :param person_count:
    :param encodings:
    :param image_name:
    :param max_len_closest_matches:
    :param start_time:
    :param i_max:
    :return:
    """
    closest_matches = []
    i = 1
    j = 1

    for folder in os.listdir(lfw_path):
        for file in os.listdir(os.path.join(lfw_path, folder)):
            filename, distance = compare_face_with_original(folder, file, encodings, file_count, person_count,
                                                            image_name, j, i)

            if len(closest_matches) < max_len_closest_matches:
                closest_matches, closest_matches_distances, index_min = add_face_to_closest_matches(filename, distance,
                                                                                                    closest_matches)
            else:
                closest_matches, closest_matches_distances, index_min = add_face_to_closest_matches_if_closer_than_current_closest_mathes(
                    filename, distance, closest_matches, closest_matches_distances, index_min)

            print_update(image_name, closest_matches, index_min, start_time, file_count, j)

            j += 1
        i += 1
        if i > i_max:
            break

    closest_matches_sorted = sorted(closest_matches, key=itemgetter(1))

    return closest_matches_sorted


def print_results(closest_matches_sorted, image_path):
    """
    
    :param closest_matches_sorted: 
    :param image_path: 
    :return: 
    """
    print("#\nRESULTS\n#\n")

    print('Closest matches to ' + image_path + ":\n")

    for result in closest_matches_sorted:
        print(result[0], ": " + str(round(result[1], 3)))


def generate_runstr(start_time, image_name, max_len_closest_matches, file_count, person_count):
    """

    :param start_time:
    :param image_name:
    :param max_len_closest_matches:
    :param file_count:
    :param person_count:
    :return:
    """

    runstr = start_time.strftime("%Y_%m_%d__%H_%M") + '_run_top_' + str(max_len_closest_matches) + \
             "_lookalikes_for_" + image_name.replace(".", "_") + "_from_" + str(file_count) + "_images_of_" + str(
        person_count) + "_people"

    return runstr


def output_results_csv(closest_matches_sorted, start_time, image_name, max_len_closest_matches, file_count,
                       person_count):
    """

    :param closest_matches_sorted:
    :return:
    """
    df = pd.DataFrame(closest_matches_sorted, columns=['person', 'facial distance'])
    df.to_csv(os.path.join(os.path.join(data_path, 'results'),
                           generate_runstr(start_time, image_name, max_len_closest_matches, file_count,
                                           person_count) + '.csv'), index=False)


def ouput_results_image(closest_matches_sorted, start_time, image_name, max_len_closest_matches, file_count,
                        person_count):
    """

    :param closest_matches_sorted:
    :param image_name:
    :return:
    """
    image_paths = [os.path.join(lfw_path, x[0]) for x in closest_matches_sorted]

    original = cv2.imread(os.path.join(os.path.join(data_path, 'inputs'), image_name))
    original = cv2.resize(original, (250, 250))

    images = [original] + [cv2.imread(x) for x in image_paths]
    combined = np.concatenate(images, axis=1)
    cv2.imwrite(os.path.join(os.path.join(data_path, 'results'),
                             generate_runstr(start_time, image_name, max_len_closest_matches, file_count,
                                             person_count) + '.jpg'), combined)


def get_start_time():
    """

    :return:
    """
    start_time = datetime.datetime.now()
    return start_time
