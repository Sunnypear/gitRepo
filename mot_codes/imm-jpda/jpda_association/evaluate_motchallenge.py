# vim: expandtab:ts=4:sw=4
import argparse
import os
import deep_sort_app


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        default='./MOT16/train/')
    parser.add_argument(
        "--detection_dir", help="Path to detections.", default='./resources/detections/MOT16_POI_train/')
    parser.add_argument(
        "--output_dir", help="Folder in which the results will be stored. Will "
        "be created if it does not exist.", default="./tmp/jpdaTrack/data/")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.3, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.0)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    return parser.parse_args()


import motmetrics as mm
import os

if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = os.listdir(args.mot_dir)
    for sequence in sequences:
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(args.mot_dir, sequence)
        detection_file = os.path.join(args.detection_dir, "%s.npy" % sequence)
        output_file = os.path.join(args.output_dir, "%s.txt" % sequence)
        deep_sort_app.run(
            sequence_dir, detection_file, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, display=False)

    '''
    gts     # gt文件列表
    results # test文件列表
    names   # 评估名称列表
    '''
    metrics = list(mm.metrics.motchallenge_metrics)
    mh = mm.metrics.create()
    accs = []
    for res_file, gt_file in zip(os.listdir(args.output_dir), os.listdir(args.mot_dir)):
    # for gt_file, result_file in zip(gts, results):
        tmp_gt = os.path.join(args.mot_dir,os.path.join(gt_file,'det\\det.txt'))
        gt = mm.io.loadtxt(tmp_gt, fmt='mot16', min_confidence=0)
        ts = mm.io.loadtxt(os.path.join(args.output_dir,res_file), fmt='mot16')
        acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
        accs.append(acc)

    # compute_many与单个文件评估略有不同，并且参数name为names
    summary = mh.compute_many(accs, metrics=metrics, names=None)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    with open("rtdetr-botsort.csv", "w", newline="") as csvfile:
        summary.to_csv(csvfile, index=True, header=True)

