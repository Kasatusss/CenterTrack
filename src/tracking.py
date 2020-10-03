import sys
import cv2
from tqdm import tqdm
from time import time
import numpy as np
import argparse


def write_results(frame_id, ret_obj, file_obj):
    for obj in ret_obj['results']:
        print(f'{frame_id},{obj["tracking_id"]},{obj["bbox"]}', file=file_obj)


def resize_image(img):
    return cv2.resize(img, (int(args.input_w), int(args.input_h)))


def process_video(filename, csv_output, generic_video):
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    assert success, 'Cant read video'

    if args.generic_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.generic_video, 
                              fourcc, args.save_framerate,
                              (int(args.input_w), int(args.input_h)))

    count, total_time = 0, []
    pbar1 = tqdm()
    with open(csv_output, 'w') as f:
        print('frame,tracking_id,bbox', file=f)
        while success:
            if count > 10:
                break
            image = resize_image(image)

            t = time()
            ret = detector.run(image)
            total_time.append(time() - t)

            write_results(count, ret, f)
            if args.generic_video:
                out.write(ret['generic'])

            count += 1
            pbar1.update(1)
            success, image = vidcap.read()

    if args.generic_video:
        out.release()
    print(f'Average inference time: {np.mean(total_time)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input data
    parser.add_argument('--video', type=str, help='path to .mp4')
    parser.add_argument('--input_h', type=str, default='512', 
                        help='input height. -1 for default from dataset.')
    parser.add_argument('--input_w', type=str, default='2048', 
                        help='input width. -1 for default from dataset.')
    parser.add_argument('--num_classes', type=str, default='1')

    # Output data
    parser.add_argument('--save_csv', type=str, help='path to create .csv file')
    parser.add_argument('--generic_video', type=str, default=None,
                        help='path to save video with tracking')
    parser.add_argument('--save_framerate', type=int, default=10,
                        help='framerate of generic video')

    # Model
    parser.add_argument('--model_path', type=str,
                        default='/content/drive/My Drive/Projects/CenterTrack/exp/tracking/football/model_last.pth',
                        help='path to model')
    parser.add_argument('--centertrack_path', type=str,
                        default='/content/drive/My Drive/Projects/CenterTrack/src/lib',
                        help='path to src/lib')

    args = parser.parse_args()


    # Import model and load weights
    sys.path.insert(0, args.centertrack_path)
    from detector import Detector
    from opts import opts
    opt = opts().init(['tracking', '--save_video', '--load_model', args.model_path,
                        '--num_classes', args.num_classes,
                        '--input_h', args.input_h, '--input_w', args.input_w,
                        '--dataset', 'custom', '--debug', '4'])
    detector = Detector(opt)


    # Process input data
    process_video(args.video, args.save_csv, args.generic_video)

