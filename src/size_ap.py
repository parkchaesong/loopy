from utils_map import compute_pre_rec, voc_ap, calc_inter_ap
import json

def calculate_ap(TEMP_FILE_PATH, results_files_path, gt_classes, opt,
                 gt_counter_per_class, gt_counter_per_size):

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

            # n_images = counter_images_per_class[class_name]
            # lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            # lamr_dictionary[class_name] = lamr

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / len(gt_classes)
        text = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)

        if not opt.no_size_ap:
            sum_AP = 0.0
            ap_dictionary = {}
            # lamr_dictionary = {}
            # open file to store the results
            results_file.write("# AP and precision/recall per size \n")

            size_count_true_positives = {}
            size_count_true_positives["small"], size_count_true_positives["medium"], \
            size_count_true_positives["large"] = 0, 0, 0
            gt_sizes = size_count_true_positives.keys()
            for size_index, size_name in enumerate(gt_sizes):
                print(size_name)
                size_tp = []
                size_fp = []
                for class_index, class_name in enumerate(gt_classes):
                    '''
                    load detection results of that class
                    '''
                    dr_file = TEMP_FILE_PATH + "/" + class_name + "_dr.json"
                    dr_data = json.load(open(dr_file))

                    '''
                    Assign detection results to gt objects
                    '''
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
                            if obj["class_name"] == class_name and obj["size_name"] == size_name:
                                bbgt = [float(x) for x in obj["bbox"].split()]
                                # 순서: left top right bottom
                                # bi = detection과 gt 중 교집합 box의 좌표
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
                            if not bool(gt_match["size_used"]):
                                # true positive
                                size_tp.append(1)
                                size_fp.append(0)
                                gt_match["size_used"] = True
                                size_count_true_positives[size_name] += 1
                                # update json file
                                with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                            else:
                                # false positive (multiple detection)
                                size_fp.append(1)
                                size_tp.append(0)
                        else:
                            size_fp.append(1)
                            size_tp.append(0)
                print(size_tp)
                rec, prec = compute_pre_rec(size_fp, size_tp, size_name, gt_counter_per_size)
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
            results_file.write("\n# mAP of all sizes\n")
            mAP = sum_AP / 3
            text = "mAP(size) = {0:.2f}%".format(mAP * 100)
            results_file.write(text + "\n")
            print(text)

            return count_true_positives, size_count_true_positives

        return count_true_positives

