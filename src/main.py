import option
import os, time, shutil
import utils_map, size_ap_v2


start = time.time()
opt = option.options

gt_json_path = opt.gt_json_path
dr_json_path = opt.dr_json_path

# if there are no classes to ignore then replace None by empty list
if opt.ignore is None:
    opt.ignore = []

# make sure that the cwd() is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

results_files_path = "results"
if not os.path.exists(results_files_path):
    os.makedirs(results_files_path)
result_file_path = results_files_path + "/results" + ".txt"

plot_result_path = "plot_figures"
if opt.draw_plot:
    if opt.plot_save:
        if os.path.exists(plot_result_path):
            shutil.rmtree(plot_result_path)
            os.makedirs(plot_result_path)
        else:
            os.makedirs(plot_result_path)

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


count_true_positives = \
    utils_map.calculate_ap(result_file_path, plot_result_path, gt_classes, opt, gt_counter_per_class, dr, gt)

size_count_true_positives = \
    size_ap_v2.calculate_ap(gt_classes, opt, dr, gt)

with open(result_file_path, 'a') as results_file:

    '''ap for classes'''
    results_file.write("\n# Number of gt objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    results_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(gt_classes):
        try: n_det = det_counter_per_classes[class_name]
        except: n_det = 0  # If there is no gt class in dt, n_dt = 0
        text = class_name + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
        results_file.write(text)

    '''ground truth & detection number for sizes'''
    results_file.write("\n# Number of gt objects per size\n")
    for class_name in gt_counter_per_size:
        results_file.write(class_name + ": " + str(gt_counter_per_size[class_name]) + "\n")

    results_file.write("\n# Number of detected objects per size\n")
    for class_name in dr_sizes:
        text = class_name + ": " + str(size_count_true_positives[class_name]) + "\n"
        results_file.write(text)

utils_map.print_configuration(result_file_path, opt)

finish = time.time()
print("time: ", finish - start)

