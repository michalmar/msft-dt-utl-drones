from PIL import Image, ImageFont, ImageDraw
import colorsys
import os
import numpy as np


# get the files to be displayed
fname = "vott-json-export/annotations.txt"
with open(fname) as f:
    lines = f.readlines()

# windows - local
lines = ["c:/Users/mimarusa/Downloads/vfn"+x.strip().replace("..","") for x in lines] 

# linux
# lines = [x.strip().replace("..","") for x in lines] 
print(len(lines))

# lines[0]

lines_valid = []
for line in lines:
    if (os.path.exists(line.split(" ")[0])):
        lines_valid.append(line)
print(len(lines_valid))


# line = '../data-in/vott-json-export/A10%20-%20Namesti.mp4#t=10927.6.jpg 248,61,443,208,0 74,90,104,113,1 637,111,657,187,2 674,140,700,214,2'
line = 'c:/Users/mimarusa/Downloads/vfn/data-in/vott-json-export/A10%20-%20Namesti.mp4#t=10927.6.jpg 248,61,443,208,0 74,90,104,113,1 637,111,657,187,2 674,140,700,214,2'


class_names = ["ER","CAR","PERSON","TRUCK"]

# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
                for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))


def display_annotations(annotation_line, class_names,colors):

    line = annotation_line
    fname = line.split(" ")[0]
    image = Image.open(fname)

    # image.show()

    boxes_raw = line.split(" ")[1:]
    boxes_tuple = [(list(map(int, b.split(",")[0:4])),int(b.split(",")[4]))  for b in boxes_raw]

    
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

    thickness = (image.size[0] + image.size[1]) // 300

    for (box,c) in boxes_tuple:
        # print(c)
        draw = ImageDraw.Draw(image)

        predicted_class = class_names[c]
        # box = out_boxes[i]
        score = 1.0

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        # x_min,y_min,x_max,y_max = box
        left,top,right,bottom = box
        # top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        # print(f'###EXPORT###{image_name}|ORIG_LABEL:{lbl_box}|{left},{top},{right},{bottom},{label}')
        # print(f'###EXPORT###{{"filename":"{image_name}","orig_label":"{lbl_box}","yolo_detection":"{left},{top},{right},{bottom}","yolo_label":"{" ".join(label.split(" ")[0:-1])}","yolo_score":"{label.split(" ")[-1]}"}}')
        # print(top, left, bottom, right, class_names[c])
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    print()
    outname = os.path.join("out","test",fname.replace(".jpg","_annotated.png"))
    image.save(outname,"PNG")
    print(f"saved to {outname}")
    # image.show()

for line in lines_valid:
    print(f'processing: {line.split(" ")[0]}')
    display_annotations(line, class_names, colors)

print("done")