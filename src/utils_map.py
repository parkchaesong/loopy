import numpy as np
import math
import os
import glob
import sys
import json
import pandas as pd
import option

opt = option.options

def calc_interpolated_prec(desired_rec, latest_pre, rec, prec):
    recall_precision = np.array([rec, prec])
    recall_precision = recall_precision.T

    inter_recall = recall_precision[recall_precision[:, 0] >= desired_rec]
    inter_precision = inter_recall[:, 1]

    if len(inter_precision) > 0:
        inter_precision = max(inter_precision)
        latest_pre = inter_precision
    else:
        inter_precision = latest_pre
    return inter_precision, latest_pre


def calc_inter_ap(opt, rec, prec):
    inter_precisions = []
    latest_pre = 0
    for i in range(opt.n_interpolation):
        recall = float(i)/(opt.n_interpolation - 1)
        inter_precision, latest_pre = calc_interpolated_prec(recall, latest_pre, rec, prec)
        inter_precisions.append(inter_precision)
    return np.array(inter_precisions).mean()


"""throw error and exit"""


def error(msg):
    print(msg)
    sys.exit(0)


"""check if the number is a float between 0.0 and 1.0"""


def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if 0.0 < val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


"""
Calculate the AP given the recall and precision array
    1) Compute version of measured precision/recall curve with precision monotonically decreasing
    2) Compute the AP as the Area Under this curve by numerical integration   ###not interpolated 
"""


def voc_ap(rec, prec):
    """official matlab code VOC2012
    mrec = [0; rec;1];
    mpre = [0; prec; 0];

    for i=numel(mpre) - 1: -1: 1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
    ap = sum((mrec(i) - mrec(i-1)).*mpre(i));
        """

    rec.insert(0, 0.0)  # insert 0.0 at beginning of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """

    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])  # 항상 i+1 의 값보다 i 가 크다 ==> 항상 감소하는 curve됨
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)  # if it was matlab would be i +1
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i-1])*mpre[i])
    return ap, mrec, mpre


"""
convert the lines of a file to a list
"""


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like '\n' at the end of each line
    content = [x.strip() for x in content]
    return content



"""
Plot - adjust axes
"""

'''
def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(rendered=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width

    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])
'''
def get_gt_match(gt_path, temp_files_path, class_dict):
    with open(gt_path) as json_file:
        json_data = json.load(json_file)
        gt_counter_per_classes = {}
        counter_images_per_classes ={}
        json_annotations = json_data["annotations"]

        gt_counter_per_sizes = {}
        counter_images_per_sizes = {}

        size_threshold = opt.size_threshold
        size_class =["small", "medium", "large"]

        json_annotations = sorted(json_annotations, key=lambda json_annotations:(json_annotations["image_id"]))
        df = pd.DataFrame(json_annotations)

        bounding_boxes = []
        already_seen_classes = []
        already_seen_sizes = []

        for idx, row in df.iterrows():

            file_id = str(row["image_id"])
            gt_id = str(row["id"])
            category_id = row["category_id"]
            class_name = class_dict[str(category_id)]
            left, top, width, height = str(row["bbox"][0]), str(row["bbox"][1]), str(row["bbox"][2]), str(
                row["bbox"][3])
            bbox = left + " " + top + " " + width + " " + height
            area = row["area"]

            if area <= (size_threshold)^2:
                size_name = "small"
            elif (size_threshold)^2 < area <= (3*size_threshold)^2:
                size_name = "medium"
            elif area > (3*size_threshold):
                size_name = "large"
            else:
                ValueError("Check the area")

            bounding_boxes.append({"file_id": file_id, "class_name":class_name, "size_name":size_name, "bbox": bbox, "used": False,
                                   "size_used": False})
            if class_name in gt_counter_per_classes:
                gt_counter_per_classes[class_name] +=1
            else:
                # if class did not exits yet
                gt_counter_per_classes[class_name] = 1

            if size_name in gt_counter_per_sizes:
                gt_counter_per_sizes[size_name] += 1
            else:
                gt_counter_per_sizes[size_name] = 1

            if class_name not in already_seen_classes:
                # 하나의 image 안에서 각 클래스가 몇번 나왔는지 계산
                if class_name in counter_images_per_classes:
                    counter_images_per_classes[class_name] += 1
                else:
                    # if class did not exist yet
                    counter_images_per_classes[class_name] = 1
                already_seen_classes.append(class_name)

            if size_name not in already_seen_sizes:
                if size_name in counter_images_per_sizes:
                    counter_images_per_sizes[size_name] +=1
                else:
                    counter_images_per_sizes[size_name] = 1
                already_seen_sizes.append(size_name)

        with open(temp_files_path + '/gt_match.json', 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    return gt_counter_per_classes, counter_images_per_classes, gt_counter_per_sizes, counter_images_per_sizes

def get_gt_lists_old(GT_PATH, TEMP_FILES_PATH, class_dict):
    with open(GT_PATH) as json_file:
        json_data = json.load(json_file)
        gt_counter_per_class = {}
        counter_images_per_classes = {}
        json_annotations = json_data["annotations"]

        gt_counter_per_size = {}
        counter_images_per_size = {}

        size_threshold = opt.size_threshold
        size_class_dict = {}
        size_class_dict["small"] = 0
        size_class_dict["medium"] = size_threshold
        size_class_dict["large"] = 3 * size_threshold

        json_annotations = sorted(json_annotations, key=lambda json_annotations: (json_annotations['image_id']))
        df = pd.DataFrame(json_annotations)
        file_id = str(df["image_id"][0])
        bounding_boxes = []
        already_seen_classes = []
        already_seen_sizes = []
        for idx, row in df.iterrows():
            # gt 파일명에 따라 이 부분 수정해야 함
            new_file_id = str(row["image_id"])
            category_id = row["category_id"]
            class_name = class_dict[str(category_id)]
            if new_file_id != file_id:
                with open(TEMP_FILES_PATH + "/" + file_id+"_ground_truth.json", "w") as outfile:
                    json.dump(bounding_boxes, outfile)
                bounding_boxes = []
            # create gt dictionary
            left, top, width, height = str(row["bbox"][0]), str(row["bbox"][1]), str(row["bbox"][2]), str(
                row["bbox"][3])
            bbox = left + " " + top + " " + width + " " + height
            area = row["bbox"][2] * row["bbox"][3]
            if area <= size_threshold * size_threshold:
                size_name = "small"
            elif size_threshold*size_threshold < area <= (3 * size_threshold) ^ 2:
                size_name = "medium"
            elif area > (3 * size_threshold) ^ 2:
                size_name = "large"
            else:
                ValueError
            bounding_boxes.append({"class_name": class_name,
                                   "size_name": size_name, "bbox": bbox, "used":False, "size_used": False})
            # count that object in all data set
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class did not exits yet
                gt_counter_per_class[class_name] = 1

            if size_name in gt_counter_per_size:
                gt_counter_per_size[size_name] += 1
            else:
                gt_counter_per_size[size_name] = 1

            if class_name not in already_seen_classes:
                # 하나의 image 안에서 각 클래스가 몇번 나왔는지 계산
                if class_name in counter_images_per_classes:
                    counter_images_per_classes[class_name] += 1
                else:
                    # if class did not exist yet
                    counter_images_per_classes[class_name] = 1
                already_seen_classes.append(class_name)

            if size_name not in already_seen_sizes:
                if size_name in counter_images_per_size:
                    counter_images_per_size[size_name] +=1
                else:
                    counter_images_per_size[size_name] = 1
                already_seen_sizes.append(size_name)
            file_id = str(row['image_id'])
        with open(TEMP_FILES_PATH + "/gt_match.json", "w") as outfile:
            json.dump(bounding_boxes, outfile)

    return gt_counter_per_class, counter_images_per_classes, gt_counter_per_size, counter_images_per_size


def check_format_class_iou(opt, gt_classes):
    n_args = len(opt.set_class_iou)
    error_msg = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)

    specific_iou_classes = opt.set_class_iou[::2]  # even elements
    iou_list = opt.set_class_iou[1::2]  # odd elements
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
            error('Error, unknown class \"'+tmp_class + '\".Flag usage:' + error_msg)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('Error, IOU must be between 0 and 1. Flag usage:' + error_msg)


def make_gt_list(gt_json_path):
    class_dict = dict()
    with open(gt_json_path) as json_file:
        json_data = json.load(json_file)
        json_categories = json_data["categories"]
        category_df = pd.DataFrame(json_categories)

        for idx, row in category_df.iterrows():
            category_id = str(row["id"])
            category_name = row.get("name")
            class_dict[category_id] = category_name
    return class_dict


def dr_json(dr_json_path, temp_file_path, class_dict):
    det_counter_per_classes = {}
    with open(dr_json_path) as origin_dr_path:
        json_data = json.load(origin_dr_path)
        json_annotations = json_data["annotations"]
        json_annotations = sorted(json_annotations, key=lambda json_annotations:(json_annotations['category_id']))
        df = pd.DataFrame(json_annotations)
        for key, value in class_dict.items():
            bounding_boxes = []
            for idx, row in df.iterrows():
                image_id = str(row['image_id'])
                #temp_path = os.path.join(gt_path, (image_id) + '.txt')
                if str(row['category_id']) == str(key):
                    tmp_class_name, confidence = value, row['score']
                    left, top, width, height = str(row["bbox"][0]), str(row["bbox"][1]), str(row["bbox"][2]), str(row["bbox"][3])
                    if tmp_class_name in det_counter_per_classes:
                        det_counter_per_classes[tmp_class_name] +=1
                    else:
                        det_counter_per_classes[tmp_class_name] = 1
                    bbox = left + " " + top + " " + width + " " + height
                    bounding_boxes.append({"confidence": confidence, "file_id": image_id, "bbox": bbox})
            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse = True)
            with open(temp_file_path + '/' + value + '_dr.json', 'w') as outfile:
                json.dump(bounding_boxes, outfile)

    return det_counter_per_classes


'''
def load_dr_into_json(GT_PATH, dr_files_list, TEMP_FILE_PATH, class_dict):
    for class_index, class_name in enumerate(class_dict):
        bounding_boxes = []
        for txt_file in dr_files_list:
            # the first time it checks if all the corresponding ground truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, width, height = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    # match
                    bbox = left + " " + top + " " + width + " " + height
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILE_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)
'''


def compute_pre_rec(fp, tp, class_name, gt_counter_per_class):
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
    # fp와 tp의 리스트를 fp인지 아닌지 반영하는 0 or 1이 아니고, 해당 인덱스까지 fp가 몇번 나왔는지 누적값으로 만든다.
    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx] / gt_counter_per_class[class_name])
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx] / (fp[idx] + tp[idx]))

    return rec, prec


def calculate_ap(temp_file_path, results_file_path, gt_classes, opt, gt_counter_per_class, counter_images_per_class):

    specific_iou_flagged = False
    if opt.set_class_iou is not None:
        specific_iou_flagged = True

    sum_AP = 0.0
    ap_dictionary = {}
    # lamr_dictionary = {}
    # open file to store the results
    with open(results_file_path + "/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class \n")
        count_true_positives = {}

        for class_indeex, class_name in enumerate(gt_classes):
            count_true_positives[class_name] =0

            dr_file = temp_file_path + '/' + class_name + '_dr.json'
            dr_data = json.load(open(dr_file))

            nd = len(dr_data)
            tp = [0] *nd
            fp = [0] * nd
            gt_file = temp_file_path + '/gt_match.json'

            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                ground_truth_data = json.load((open(gt_file)))
                ovmax = -1
                gt_match = -1
                bb = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    if not obj["file_id"] == file_id:
                        pass
                    else:
                        if obj["class_name"] == class_name:
                            bbgt = [float(x) for x in obj["bbox"].split()]
                            bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[0] + bb[2], bbgt[0] + bbgt[2]),
                                  min(bb[1] + bb[3], bbgt[1] + bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # ua = compute overlap (IoU) = area of intersection/ area of union
                                ua = ((bb[2] + 1) * (bb[3] + 1) + (bbgt[2] + 1) * (bbgt[3] + 1)) - iw * ih
                                IoU = iw * ih / ua
                                if IoU > ovmax:
                                    ovmax = IoU
                                    gt_match = obj

                iou_threshold = opt.iou_threshold
                if specific_iou_flagged:
                    specific_iou_classes = opt.set_class_iou[::2]
                    iou_list = opt.set_class_iou[1::2]
                    if class_name in specific_iou_classes:
                        index = specific_iou_classes.index(class_name)
                        IoU_threshold = float(iou_list[index])
                if ovmax >= iou_threshold:
                    if not bool(gt_match["used"]):
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] +=1
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        fp[idx] = 1
                else:
                    fp[idx] = 1

            rec, prec = compute_pre_rec(fp, tp, class_name, gt_counter_per_class)
            if opt.no_interpolation:
                ap, mrec, mprec = voc_ap(rec[:], prec[:])
            else:
                ap = calc_inter_ap(opt, rec[:], prec[:])
            # ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP += ap
            text = "{0:.2f}%".format(
                ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(
                text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")

            if not opt.quiet:
                print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / len(gt_classes)
        text = "mAP = {0:.2f}%".format(mAP * 100)
        results_file.write(text + "\n")
        print(text)
    return count_true_positives


def calculate_ap_old(TEMP_FILE_PATH, results_files_path, gt_classes, opt,
                 gt_counter_per_class, counter_images_per_class):

    specific_iou_flagged = False
    if opt.set_class_iou is not None:
        specific_iou_flagged = True

    sum_AP = 0.0
    ap_dictionary = {}
    # lamr_dictionary = {}
    # open file to store the results
    with open(results_files_path + "/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class \n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            '''
            load detection results of that class
            '''
            dr_file = TEMP_FILE_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            '''
            Assign detection results to gt objects
            '''
            nd = len(dr_data)
            tp = [0] * nd  # nd 사이즈만큼 zero array 생성
            fp = [0] * nd

            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                # assign detection results to gt object if any
                # open gt with that file id
                gt_file = TEMP_FILE_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [float(x) for x in detection["bbox"].split()]
                confidence = float(detection["confidence"])

                # ==> confidence 부분 다시 생각해봐야함
                # if confidence < opt.confidence_threshold:
                #    fp[idx] = 1
                #    continue
                for obj in ground_truth_data:
                    # look for class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        # 순서: left top right bottom
                        # bi = detection과 gt 중 교집합 box의 좌표
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[0] + bb[2], bbgt[0]+bbgt[2]),
                              min(bb[1] + bb[3], bbgt[1] + bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # ua = compute overlap (IoU) = area of intersection/ area of union
                            ua = ((bb[2] + 1) * (bb[3] + 1) + (bbgt[2] + 1) * (bbgt[3] + 1)) - iw*ih
                            IoU = iw * ih / ua
                            if IoU > ovmax:
                                ovmax = IoU
                                gt_match = obj
                # assign detection as true positive/false positive

                # set minimum overlap threshold
                IoU_threshold = opt.iou_threshold
                if specific_iou_flagged:
                    specific_iou_classes = opt.set_class_iou[::2]
                    iou_list = opt.set_class_iou[1::2]
                    if class_name in specific_iou_classes:
                        index = specific_iou_classes.index(class_name)
                        IoU_threshold = float(iou_list[index])
                if ovmax >= IoU_threshold:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update json file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                else:
                    fp[idx] = 1

            rec, prec = compute_pre_rec(fp, tp, class_name, gt_counter_per_class)
            if opt.no_interpolation:
                ap, mrec, mprec = voc_ap(rec[:], prec[:])
            else:
                ap = calc_inter_ap(opt, rec[:], prec[:])
            # ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP += ap
            text = "{0:.2f}%".format(
                ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")

            if not opt.quiet:
                print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / len(gt_classes)
        text = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)
    return count_true_positives



