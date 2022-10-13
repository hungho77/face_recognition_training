import os
import sys
import argparse

from PIL import Image
import cv2
import numpy as np
from skimage import transform as trans

from face_alignment.mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from face_alignment.mtcnn_pytorch.src import detect_faces, show_bboxes

src = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041] ], dtype=np.float32 )

def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path = args.input_dir
    
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    minsize = 20 # minimum size of face

    for root, dirs, files in os.walk(src_path):
        for img_name in files:
            nrof_images_total+=1
            path = os.path.join(root, img_name)
            img = Image.open(path)
            img_cv2 = np.array(img)[...,::-1]
            if not os.path.exists(os.path.join(output_dir, path.split("/")[-2])):
                os.makedirs(os.path.join(output_dir, path.split("/")[-2]))
            try:
                
                bounding_boxes, landmarks = detect_faces(img)
                box = bounding_boxes[0][:4].astype(np.int32).tolist()
                dst = landmarks[0].astype(np.float32)
                facial5points = [[dst[j],dst[j+5]] for j in range(5)]
                face = img_cv2[box[1]:box[3],box[0]:box[2]]
                landmark = landmarks[0]
                facial5points = [[landmark[j] - box[0],landmark[j+5] - box[1]] for j in range(5)]
                reference_pts = get_reference_facial_points(default_square= True)
                dst_img = warp_and_crop_face(face, facial5points, reference_pts, crop_size=(112,112))
                save_path = os.path.join(output_dir, "/".join(path.split("/")[-2:]))
                cv2.imwrite(save_path, dst_img)
                nrof_successfully_aligned+=1
                
            except Exception as e:
                print(e)
                print(path)
                                          
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', default="./FSSProfileHCM", type=str, help='Directory with unaligned images.')
    parser.add_argument('--output_dir', default="./aligned_FSSProfileHCM", type=str, help='Directory with aligned face thumbnails.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))