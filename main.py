from configs import lfw_path
from utils import (
    get_numbers_of_images,
    get_number_of_pics_to_compare_with,
    get_image_encodings,
    compare_with_other_images,
    print_results,
    output_results_csv,
    output_results_image,
    get_start_time,
)
import argparse


def run(n_people_to_compare_with, max_len_closest_matches):
    start_time = get_start_time()

    (
        n_people_to_compare_with,
        image_name,
        person_count,
        image_path,
    ) = get_numbers_of_images(max_len_closest_matches, n_people_to_compare_with)

    file_count = get_number_of_pics_to_compare_with(
        n_people_to_compare_with, lfw_path, person_count, image_name
    )

    original_image_encodings = get_image_encodings(image_path)

    closest_matches_sorted = compare_with_other_images(
        lfw_path,
        file_count,
        original_image_encodings,
        image_name,
        max_len_closest_matches,
        start_time,
        n_people_to_compare_with,
    )

    print_results(closest_matches_sorted, image_path)

    output_results_csv(
        closest_matches_sorted,
        start_time,
        image_name,
        max_len_closest_matches,
        file_count,
        n_people_to_compare_with,
    )

    output_results_image(
        closest_matches_sorted,
        start_time,
        image_name,
        max_len_closest_matches,
        file_count,
        n_people_to_compare_with,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--n_people_to_compare_with",
        type=int,
        required=False,
        help="How many people's faces do you want to compare with? (default all)",
    )
    parser.add_argument(
        "-m",
        "--max_len_closest_matches",
        type=int,
        default=10,
        help="How many closest matches do you want to see? (default 10)",
    )
    args = parser.parse_args()

    run(
        n_people_to_compare_with=args.n_people_to_compare_with,
        max_len_closest_matches=args.max_len_closest_matches,
    )
