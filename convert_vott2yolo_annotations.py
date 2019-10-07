###############################################################################
# Michal Marusan
#
# Usage:
#   python convert_vott2yolo_annotations.py --dataset_dir data  --annotation_file DOTS-V2-export-Jakub.json --prj_upd
###############################################################################
import json
import os
import random
# import azureml.core
import argparse
import shutil



def convert_vott2yolo(dataset_dir, annotation_file, update_project_dir = False):
    annotations = json.load(open(os.path.join(dataset_dir, annotation_file)))

    clss = []
    with open(os.path.join(dataset_dir, "classes.txt"), "w") as f:
        for tid, t in enumerate(annotations["tags"]):
            # Add classes. We have only one class to add.
            # self.add_class("gate_control", tid, t["name"])
            clss.append(t["name"])
            f.write(t["name"]+"\n")
    i = 0
    annotation_rows = []
    
    for key, asset_tmp in annotations["assets"].items():
        i+=1
        # if (i>500): 
        #     break
        w,h = (asset_tmp["asset"]["size"]["width"], asset_tmp["asset"]["size"]["height"])
        img_name = asset_tmp["asset"]["name"]
        image_path = os.path.join(dataset_dir, img_name)

        annotation_row = f'{image_path}'

        regs = []
        tags = []
        has_regions = False
        for ar in asset_tmp["regions"]:
            reg_type = ar["type"]
            reg_tags = ar["tags"]
            reg_bb = ar["boundingBox"]
            r = {}
            r_all_points_x = []
            r_all_points_y = []
            for p in ar["points"]:
                r_all_points_x.append(p["x"])
                r_all_points_y.append(p["y"])
            r["shape_attributes"] = {"all_points_x": r_all_points_x, "all_points_y": r_all_points_y}
            regs.append(r)
            tags.append(reg_tags)
            has_regions = True

            # Row format: image_file_path box1 box2 ... boxN;
            # Box format: x_min,y_min,x_max,y_max,class_id (no space).
            annotation_row += f' {int(r_all_points_x[0])},{int(r_all_points_y[0])},{int(r_all_points_x[2])},{int(r_all_points_y[2])},{clss.index(ar["tags"][0])}'
        
        annotation_row += "\n"
        if (has_regions):
            annotation_rows.append(annotation_row)

    # division to TRAIN and TEST (random)
    random.shuffle(annotation_rows)
    split_ratio = 0.8
    split_index = int(len(annotation_rows) * split_ratio)
    annotation_rows_train = annotation_rows[:split_index]
    annotation_rows_test = annotation_rows[split_index:]
    
    with open(os.path.join(dataset_dir, "annotations.txt"), "w") as f:
        for annotation_row in annotation_rows_train:
            f.write(annotation_row)

    with open(os.path.join(dataset_dir, "annotations_test.txt"), "w") as f:
        for annotation_row in annotation_rows_test:
            f.write(annotation_row)

    if (update_project_dir):
        shutil.copyfile(os.path.join(dataset_dir, "classes.txt"), os.path.join("aml_prj", "model_data","classes.txt"))
        shutil.copyfile(os.path.join(dataset_dir, "annotations.txt"), os.path.join("aml_prj", "model_data","annotations.txt"))
        shutil.copyfile(os.path.join(dataset_dir, "annotations_test.txt"), os.path.join("aml_prj", "model_data","annotations_test.txt"))
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AML Service Workspace setup.',argument_default=argparse.SUPPRESS)
    parser.add_argument('--dataset_dir', type=str, dest='dataset_dir', help='dataset_dir...', default="data")
    parser.add_argument('--annotation_file', type=str, dest='annotation_file', help='annotation_file...', default="xxx.json")
    parser.add_argument('--prj_upd', help='update also aml project', action='store_true')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    annotation_file = args.annotation_file

    try:
        if (args.prj_upd):
            prj_upd = True
    except:
        prj_upd = False

    # convert_vott2yolo("vott-json-export", "VFN-entry-gate-vid-export.json")
    # convert_vott2yolo("vott-json-export-20190618", "VFN-Gate-Control-short-Videos-Images-only-export.json")

    ## new annotation from Filip based on new videos (202)
    # convert_vott2yolo("vott-json-export-20190808", "VFN-entry-gate-vid-export.json")
    # annotations from images only
    convert_vott2yolo(dataset_dir, annotation_file, )