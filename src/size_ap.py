from utils_map import compute_pre_rec, voc_ap, calc_inter_ap
import json

def calculate_ap(results_file_path, gt_classes, opt, gt_counter_per_class, dr, gt):
    specific_iou_flagged = False
    if opt.set_class_iou is not None:
        specific_iou_flagged = True
    sum_AP = 0.0
    ap_dictionary = {}
    with open(results_file_path + "/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class \n")
        count_true_positives = {}
        size_dict = {}
        size_dict["small"], size_dict["medium"], size_dict["large"] = [], [], []
        for idx, scale in enumerate(size_dict):
            count_true_positives[scale] = 0
            tp = []
            fp = []
            for class_index, class_name in enumerate(gt_classes):
                dr_data = dr[class_name]
                gt_data = gt[class_name]
                for idx, detection in enumerate(dr_data):
                    file_id = detection["file_id"]
                    ovmax = -1
                    gt_match = -1
                    bb = [float(x) for x in detection["bbox"].split()]
                    for idx, obj in enumerate(gt_data):
                        if obj["size_name"] == scale and obj["file_id"] == file_id:
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
                            index = specific_iou_classes.index(scale)
                            iou_threshold = float(iou_list[index])
                    if ovmax >= iou_threshold:
                        if not bool(gt_match["used"]):
                            tp.append(1)
                            fp.append(0)
                            gt_match["used"] = True
                            count_true_positives[scale] +=1
                            obj = gt_match

                        else:
                            fp.append(1)
                            tp.append(0)
                    else:
                        fp.append(1)
                        tp.append(0)
            rec, prec = compute_pre_rec(fp, tp, scale, gt_counter_per_class)
            if opt.no_interpolation:
                ap, mrec, mprec = voc_ap(rec[:], prec[:])
            else:
                ap = calc_inter_ap(opt, rec[:], prec[:])
            sum_AP += ap
            text = "{0:.2f}%".format(
                ap * 100) + " = " + scale + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(
                text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")

            if not opt.quiet:
                print(text)
            ap_dictionary[scale] = ap
        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / 3
        text = "mAP = {0:.2f}%".format(mAP * 100)
        results_file.write(text + "\n")
        print(text)
    return count_true_positives

