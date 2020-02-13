import option
import os
import shutil
import utils_map
import size_ap
import time

main_start = time.time()
opt = option.options

gt_json_path = 'C:/Users/장용원/Desktop/sphinx/sphinx/test.json'
dr_json_path = 'C:/Users/장용원/Desktop/sphinx/sphinx/output.json'

# if there are no classes to ignore then replace None by empty list
if opt.ignore is None:
    opt.ignore = []

# make sure that the cwd() is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

results_files_path = "results"
if os.path.exists(results_files_path):
    shutil.rmtree(results_files_path)
else:
    os.makedirs(results_files_path)

class_dict = utils_map.make_gt_list(gt_json_path)
gt_counter_per_class, counter_images_per_class, gt_counter_per_size, counter_images_per_size, gt \
    = utils_map.get_gt_match(gt_json_path, class_dict)


gt_classes = list(class_dict.values())
# sort classes alphabetically

gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)

if opt.set_class_iou is not None:
    utils_map.check_format_class_iou(opt, gt_classes)
det_counter_per_classes, dr = utils_map.dr_json(dr_json_path, class_dict)
dr_classes = list(det_counter_per_classes.keys())
dr_sizes = ["small", "medium", "large"]
if opt.no_size_ap:
    count_true_positives = utils_map.calculate_ap(results_files_path, gt_classes, opt,
                                                gt_counter_per_class, dr, gt)

    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of gt objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_classes[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            results_file.write(text)

    finish = time.time()
    print("time: ", finish - main_start)
else:

    count_true_positives \
        = size_ap.calculate_ap(results_files_path, gt_classes, opt, gt_counter_per_size, dr, gt)
    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of gt objects per size\n")
        for class_name in sorted(gt_counter_per_size):
            results_file.write(class_name + ": " + str(gt_counter_per_size[class_name]) + "\n")


