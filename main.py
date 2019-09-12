from configs import lfw_path
from utils import get_user_inputs, get_number_of_pics_to_compare_with, get_image_encodings, \
    compare_with_all_other_images, print_results, output_results_csv, ouput_results_image, get_start_time


def run():
    start_time = get_start_time()

    i_max, max_len_closest_matches, image_name, person_count, image_path = get_user_inputs()

    file_count = get_number_of_pics_to_compare_with(i_max, lfw_path, person_count, image_name)

    original_image_encodings = get_image_encodings(image_path)

    closest_matches_sorted = compare_with_all_other_images(lfw_path, file_count, person_count, original_image_encodings,
                                                           image_name, max_len_closest_matches, start_time, i_max)

    print_results(closest_matches_sorted, image_path)

    output_results_csv(closest_matches_sorted, start_time, image_name, max_len_closest_matches, file_count,
                       person_count)

    ouput_results_image(closest_matches_sorted, start_time, image_name, max_len_closest_matches, file_count,
                        person_count)


if __name__ == "__main__":
    run()
