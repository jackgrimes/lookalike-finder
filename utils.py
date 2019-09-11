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

    potential_images = [x for x in os.listdir(os.path.join(data_path, 'inputs')) if x.split('.')[1] in ALLOWED_PICTURE_FILE_TYPES]

    for x in potential_images:
        response = input("Comparing " + x + "? (y for yes, n or no)")
        while response not in ['y', 'n']:
            print("enter y or n")
            response = input("Comparing " + x + "? (y for yes, n or no)")
        if response == "y":
            image_name = x
            break

    return i_max, max_len_closest_matches, image_name, person_count
