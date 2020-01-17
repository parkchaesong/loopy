import json
import os
import pandas as pd


gt_json_path = 'C:/Users/장용원/Desktop/loopy-master/loopy-master/gt.json'
dr_json_path = 'C:/Users/장용원/Desktop/loopy-master/loopy-master/pred.json'

gt_txt_path = 'C:/Users/장용원/Desktop/loopy-master/loopy-master/gt'
dr_txt_path = 'C:/Users/장용원/Desktop/loopy-master/loopy-master/dr'

if not os.path.isdir(gt_txt_path):
    os.makedirs(gt_txt_path)
if not os.path.isdir(dr_txt_path):
    os.makedirs(dr_txt_path)

category_dict = {}
with open(gt_json_path) as json_file:
    json_data = json.load(json_file)

    json_images_list = json_data["images"]
    json_categories = json_data["categories"]
    json_annotations = json_data["annotations"]
    json_annotations = sorted(json_annotations, key=lambda json_annotations: (json_annotations['image_id']))
    # image id 대로 sorting
    category_df = pd.DataFrame(json_categories)
    df = pd.DataFrame(json_annotations)
    category_dict = {}
    '''
    for json_image in json_images_list:
        for key, value in enumerate(json_image):
            if value == "id":
                file_name = str(json_image["id"]) + '.txt'
                txt = open(file_name, 'w')
                data = ('image_id: %d \n' % json_image["id"])
                txt.write(data)
                txt.close()
            else:
                ValueError("pass")'''
    for idx, row in category_df.iterrows():
        category_id = str(row["id"])
        category_name = row.get("name")
        category_dict[category_id] = category_name

    for idx, row in df.iterrows():
        data = str(row["image_id"])
        file_name = gt_txt_path + '/' + data + '_gt' + '.txt'
        '''if not os.path.isfile(file_name):

            txt = open(file_name, 'w')
            data_id = ('image_id: %d \n' % row["image_id"])
            txt.write(data_id)
            txt.close()'''

        txt = open(file_name, 'a')
        data_cid = row["category_id"]
        data_cname = category_dict.get(str(data_cid))
        txt.write(data_cname)
        data_bbx_x_center, data_bbx_y_center, data_bbx_w, data_bbx_h =\
            str(row["bbox"][0]), str(row["bbox"][1]), str(row["bbox"][2]), str(row["bbox"][3])
        # txt.write('data_bbx: ')
        txt.write(' ' + data_bbx_x_center + ', ' + data_bbx_y_center + ', ' + data_bbx_w + ', ' + data_bbx_h)
        txt.close()

with open(dr_json_path) as json_file:
    json_data = json.load(json_file)

    json_categories = json_data["categories"]
    json_annotations = json_data["annotations"]
    json_annotations = sorted(json_annotations, key=lambda json_annotations: (json_annotations['image_id']))
    # image id 대로 sorting, image_id ==> score 로 바꾸면 confidence sorting
    df = pd.DataFrame(json_annotations)
    for idx, row in df.iterrows():
        data = str(row["image_id"])

        file_name = dr_txt_path + '/'+ data + '_dr' + '.txt'
        '''if not os.path.isfile(file_name):
            txt = open(file_name, 'w')
            data_id = ('image_id: %d \n' % row["image_id"])
            txt.write(data_id)
            txt.close()
            '''
        txt = open(file_name, 'a')
        data_cid = row["category_id"]
        data_cname = category_dict.get(str(data_cid))
        txt.write(data_cname)
        data_bbx_x_center, data_bbx_y_center, data_bbx_w, data_bbx_h =\
            str(row["bbox"][0]), str(row["bbox"][1]), str(row["bbox"][2]), str(row["bbox"][3])
        #txt.write('pred_bbx: ')
        data_score = str(row["score"])
        txt.write(' ' + data_score)
        txt.write(' ' + data_bbx_x_center + ' ' + data_bbx_y_center + ' ' + data_bbx_w + ' ' + data_bbx_h + '\n')
        txt.close()







