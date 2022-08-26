import datetime
import itertools
import logging
import os
import re

import cv2
import face_recognition
import numpy as np
import pandas as pd
from tqdm import tqdm

from configs import lfw_path, data_path, ALLOWED_PICTURE_FILE_TYPES

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.DEBUG,
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
logger = logging.getLogger(__name__)

tqdm.pandas()


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
        return (
            str(hours)
            + " hours, "
            + str(minutes)
            + " minutes, "
            + str(seconds)
            + " seconds taken so far"
            + "\n"
            + "Estimated "
            + str(remaining_hours)
            + " hours, "
            + str(remaining_minutes)
            + " minutes, "
            + str(remaining_seconds)
            + " seconds to completion"
        )
    except Exception:
        return "Cannot calculate times done and remaining at this time"


def choose_input_image(potential_images):
    """

    :param potential_images:
    :return:
    """
    for x in itertools.cycle(potential_images):
        response = input("Comparing " + x + "? (y for yes, n or no)")
        while response not in ["y", "n"]:
            print("enter y or n")
            response = input("Comparing " + x + "? (y for yes, n or no)")
        if response == "y":
            image_name = x
            break

    print("")

    return image_name


def get_numbers_of_images(max_len_closest_matches, n_people_to_compare_with):
    """

    :return:
    """
    if max_len_closest_matches == "":
        max_len_closest_matches = 10

    person_count = len(([len(files) for r, d, files in os.walk(lfw_path)])) - 1

    if n_people_to_compare_with is None:
        n_people_to_compare_with = person_count

    n_people_to_compare_with = int(n_people_to_compare_with)

    return n_people_to_compare_with, person_count


def get_number_of_pics_to_compare_with(
    n_people_to_compare_with, lfw_path, person_count
):
    """

    :param n_people_to_compare_with:
    :param lfw_path:
    :return:
    """

    file_count = 0
    person_no = 0
    for root, dirs, files in os.walk(lfw_path):
        if person_no <= n_people_to_compare_with:
            person_no += 1
            file_count += len([x for x in files if x != "Thumbs.db"])
        else:
            break

    if n_people_to_compare_with < person_count:
        person_count = n_people_to_compare_with
        file_count = sum(
            [len(files) for r, d, files in os.walk(lfw_path)][
                1 : (n_people_to_compare_with + 1)
            ]
        )
        logger.debug(
            str(file_count)
            + " images of "
            + str(person_count)
            + " people to compare with "
        )

    return file_count


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
    index_min = min(
        range(len(closest_matches_distances)), key=closest_matches_distances.__getitem__
    )
    return closest_matches, closest_matches_distances, index_min


def add_face_to_closest_matches_if_closer_than_current_closest_mathes(
    filename, distance, closest_matches, closest_matches_distances, index_min
):
    """

    :param filename:
    :param distance:
    :param closest_matches:
    :param closest_matches_distances:
    :return:
    """
    if distance < max(closest_matches_distances):
        closest_matches.append([filename, round(distance, 3)])
        index_max = max(
            range(len(closest_matches_distances)),
            key=closest_matches_distances.__getitem__,
        )
        del closest_matches[index_max]
        closest_matches_distances = get_closest_matches_distances(closest_matches)
        index_min = min(
            range(len(closest_matches_distances)),
            key=closest_matches_distances.__getitem__,
        )

    return closest_matches, closest_matches_distances, index_min


def print_update(
    image_name, closest_matches, index_min, start_time, file_count, image_counter
):
    """

    :param image_name:
    :param closest_matches:
    :param index_min:
    :param start_time:
    :param file_count:
    :param image_counter:
    :return:
    """
    print(
        "Current closest lookalike to "
        + image_name
        + " is "
        + closest_matches[index_min][0]
        + " with a distance of "
        + str(round(closest_matches[index_min][1], 3))
    )
    print(timesince(start_time, 100 * image_counter / file_count) + "\n")


def compare_with_other_images(
    encodings, n_people_to_compare_with, images_and_encodings
):
    """

    :param lfw_path:
    :param file_count:
    :param person_count:
    :param encodings:
    :param image_name:
    :param max_len_closest_matches:
    :param start_time:
    :param n_people_to_compare_with:
    :return:
    """

    if n_people_to_compare_with < len(images_and_encodings):
        images_and_encodings = images_and_encodings.sample(n_people_to_compare_with)

    logger.debug("Getting face distances")
    images_and_encodings["face_distance"] = images_and_encodings[
        "lfw_encodings"
    ].progress_apply(lambda x: face_recognition.face_distance(x, encodings)[0])

    images_and_encodings.sort_values(["face_distance"], inplace=True)

    return images_and_encodings


def print_results(closest_matches_sorted, image_path):
    """

    :param closest_matches_sorted:
    :param image_path:
    :return:
    """
    print("#\nRESULTS\n#\n")

    print("Closest matches to " + image_path + ":\n")

    for result in closest_matches_sorted:
        print(result[0], ": " + str(round(result[1], 3)))


def generate_runstr(
    start_time,
    image_name,
    max_len_closest_matches,
    file_count,
    n_people_to_compare_with,
):
    """

    :param start_time:
    :param image_name:
    :param max_len_closest_matches:
    :param file_count:
    :param person_count:
    :return:
    """

    runstr = (
        start_time.strftime("%Y_%m_%d__%H_%M")
        + "_run_top_"
        + str(max_len_closest_matches)
        + "_lookalikes_for_"
        + image_name.replace(".", "_")
        + "_from_"
        + str(file_count)
        + "_images_of_"
        + str(n_people_to_compare_with)
        + "_people"
    )

    return runstr


def output_results_csv(
    closest_matches_sorted,
    start_time,
    image_name,
    max_len_closest_matches,
    file_count,
    n_people_to_compare_with,
):
    """

    :param closest_matches_sorted:
    :return:
    """
    results = pd.DataFrame(
        closest_matches_sorted, columns=["person", "facial distance"]
    )
    results.to_csv(
        os.path.join(
            os.path.join(data_path, "results"),
            generate_runstr(
                start_time,
                image_name,
                max_len_closest_matches,
                file_count,
                n_people_to_compare_with,
            )
            + ".csv",
        ),
        index=False,
    )


def output_results_image(
    closest_matches_sorted,
    start_time,
    image_name,
    max_len_closest_matches,
    file_count,
    n_people_to_compare_with,
):
    """

    :param closest_matches_sorted:
    :param image_name:
    :return:
    """
    image_paths = [os.path.join(lfw_path, x[0]) for x in closest_matches_sorted]

    original = cv2.imread(os.path.join(os.path.join(data_path, "inputs"), image_name))
    original = cv2.resize(original, (250, 250))

    images = [original] + [cv2.imread(x) for x in image_paths]
    combined = np.concatenate(images, axis=1)
    cv2.imwrite(
        os.path.join(
            os.path.join(data_path, "results"),
            generate_runstr(
                start_time,
                image_name,
                max_len_closest_matches,
                file_count,
                n_people_to_compare_with,
            )
            + ".jpg",
        ),
        combined,
    )


def get_start_time():
    """

    :return:
    """
    start_time = datetime.datetime.now()
    return start_time


def get_encodings_from_path(p):
    image = face_recognition.load_image_file(p)
    encodings = face_recognition.face_encodings(image)
    return encodings


def find_best_face(person, image_path, encodings):
    if len(encodings) == 1:
        return encodings[0]
    elif len(encodings) == 0:
        return []
    else:
        if len(encodings) > 1:
            logger.info(f"Finding best face for {image_path}")
            all_images_this_person = [
                os.path.join(person, im) for im in os.listdir(person)
            ]
            if len(all_images_this_person) > 1:
                im_encodings = {}
                other_images_this_person = all_images_this_person
                other_images_this_person.remove(image_path)
                for i in other_images_this_person:
                    im_encodings[i] = get_encodings_from_path(i)

                distances = []

                for i, original_image_encodings in enumerate(encodings):
                    for other_image in other_images_this_person:
                        other_image_encodings = get_encodings_from_path(other_image)
                        distances.append(
                            (
                                i,
                                face_recognition.face_distance(
                                    other_image_encodings, original_image_encodings
                                ),
                            )
                        )

                distances_df = pd.DataFrame(distances)

                distances_df = (
                    distances_df.explode(1).groupby(0).mean().reset_index(drop=False)
                )

                best_face_index = distances_df[0][distances_df[1].idxmin()]

                return encodings[best_face_index]
            else:
                return []


def read_encodings_from_images():
    images_and_encodings = pd.DataFrame(
        [
            (r, os.path.join(r, f))
            for r, d, files in os.walk(lfw_path)
            for f in files
            if (sum(ft in f for ft in ALLOWED_PICTURE_FILE_TYPES) > 0)
        ],
        columns=["person", "image_path"],
    )

    images_and_encodings["encodings"] = images_and_encodings[
        "image_path"
    ].progress_apply(get_encodings_from_path)

    images_and_encodings["optimal_encodings"] = images_and_encodings.apply(
        lambda x: find_best_face(x.person, x.image_path, x.encodings), axis=1
    )

    images_and_encodings.drop(columns=["encodings"], inplace=True)

    images_and_encodings.rename(
        columns={"optimal_encodings": "encodings"}, inplace=True
    )

    images_and_encodings = images_and_encodings[
        images_and_encodings["encodings"].apply(len) > 0
    ]

    return images_and_encodings


def get_lfw_face_encodings():
    """
    Overall runner function
    :return:
    """

    encodings_absent = (
        len(
            [
                f
                for f in os.listdir(os.path.join(data_path, "encodings"))
                if f.endswith(".csv")
            ]
        )
        == 0
    )

    if encodings_absent:

        logger.debug("Getting encodings from images")

        images_and_encodings = read_encodings_from_images()

        images_and_encodings.to_csv(
            os.path.join(data_path, "encodings", "encodings.csv"), index=False
        )

        images_and_encodings.rename(
            columns={"encodings": "lfw_encodings"}, inplace=True
        )

    else:
        images_and_encodings = pd.read_csv(
            os.path.join(data_path, "encodings", "encodings.csv")
        )

        def convert_to_numpy(s):
            s = re.sub("\n", "", s)
            s = re.sub("[\[',\]]", "", s)
            s = re.sub(" +", " ", s)
            s = re.sub("( )*$", "", s)
            li = s.split(" ")
            li = [float(n) for n in li]
            return np.array(li)

        images_and_encodings["encodings"] = images_and_encodings["encodings"].apply(
            convert_to_numpy
        )

        images_and_encodings.rename(
            columns={"encodings": "lfw_encodings"}, inplace=True
        )

    return images_and_encodings
