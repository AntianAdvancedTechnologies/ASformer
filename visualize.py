import cv2
import argparse

import visualization.plot_segments as plot_segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help="path to desired input video file")
    parser.add_argument('--gt_file', type=str, help="path to ground truth of that video file")
    parser.add_argument('--prediction_file', type=str, help="path to prediction npy of that video file")
    parser.add_argument('--mapping_file', type=str, help="path to mapping file")
    args = parser.parse_args()

    plot_segment = plot_segments.PlotSegments(args.mapping_file)
    plot_segment.visualize(args.video_path, args.gt_file, args.prediction_file)


# python visualize.py --video_path <path_to_video> --gt_arr_file <path_to_gt_file,(.npy)> --prediction_file <path_to_prediction_file (.npy)> --mapping_file <path_to_mapping_file (.txt)>

# python visualize.py --video_path data_snatch_100-20240428T075449Z-001\data_snatch_100\Snatch_1.mp4 --gt_arr_file <path_to_gt_file,(.npy)> --prediction_file <path_to_prediction_file (.npy)> --mapping_file <path_to_mapping_file (.txt)>

