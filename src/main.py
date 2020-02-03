import option
import os
import shutil
import utils_map
import size_ap
import time

start = time.time()
opt = option.options


# if there are no classes to ignore then replace None by empty list
if opt.ignore is None:
    opt.ignore = []

# make sure that the cwd() is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))


gt_json_path = 'C:/Users/장용원/Desktop/sphinx/sphinx/test.json'
dr_json_path = 'C:/Users/장용원/Desktop/sphinx/sphinx/result.json'

# if there no images then no animation can be shown
IMG_PATH = os.path.join(os.getcwd(), 'input', 'images-optional')

'''
if os.path.exists(IMG_PATH):
    for dirpath, dirnames, files in os.walk(IMG_PATH):
        if not files:
            opt.no_animation = True
else:
    opt.no_animation = True
'''

TEMP_FILES_PATH = ".temp_files"
if not os.path.exists(TEMP_FILES_PATH):
    os.makedirs(TEMP_FILES_PATH)
results_files_path = "results"
if os.path.exists(results_files_path):
    shutil.rmtree(results_files_path)
else:
    os.makedirs(results_files_path)

class_dict = utils_map.make_gt_list(gt_json_path)

gt_counter_per_class, counter_images_per_class, gt_counter_per_size, counter_images_per_size \
    = utils_map.get_gt_match(gt_json_path, TEMP_FILES_PATH, class_dict)


gt_classes = list(class_dict.values())
# sort classes alphabetically

gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)

if opt.set_class_iou is not None:
    utils_map.check_format_class_iou(opt, gt_classes)

det_counter_per_classes = utils_map.dr_json(dr_json_path, TEMP_FILES_PATH, class_dict)

if opt.no_size_ap:
    count_true_positives = utils_map.calculate_ap(TEMP_FILES_PATH, results_files_path, gt_classes, opt,
                                                gt_counter_per_class, counter_images_per_class)
else:
    count_true_positives, size_count_true_positives \
        = size_ap.calculate_ap(TEMP_FILES_PATH, results_files_path, gt_classes, opt, gt_counter_per_class, gt_counter_per_size)

shutil.rmtree(TEMP_FILES_PATH)

'''
Count total of detection-results
'''

'''
det_counter_per_classes = {}
for txt_file in dr_files_list:
    # get lines to list
    lines_list = utils_map.file_lines_to_list(txt_file)
    for line in lines_list:
        class_name = line.split()[0]
        # check if class is in the ignore list, if yes skip
        if class_name in opt.ignore:
            continue
        if class_name in det_counter_per_classes:
            det_counter_per_classes[class_name] += 1
        else:
            # class did not exist yet
            det_counter_per_classes[class_name] = 1
'''
dr_classes = list(det_counter_per_classes.keys())


'''
Write num of gt object per classes to results.txt
'''

with open(results_files_path + "/results.txt", 'a') as results_file:
    results_file.write("\n# Number of gt objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

'''
Finish counting tp
'''

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
print("time: ", finish - start)

