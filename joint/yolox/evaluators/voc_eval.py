#!/usr/bin/env python3
# Code are based on
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# Copyright (c) Bharath Hariharan.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np


def parse_rec(filename):
    """Parse a PASCAL VOC xml file"""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects
def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations

def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    detpath,
    annopath,
    imagesetfile,
    classname,
    cachedir,
    ovthresh=0.5,
    use_07_metric=False,
):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "annots.pkl")
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print(f"Reading annotation for {i + 1}/{len(imagenames)}")
        # save
        print(f"Saving cached annotations to {cachefile}")
        with open(cachefile, "wb") as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, "rb") as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return 0, 0, 0

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0) - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

        # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate_coco_map(
    generator,
    retinanet,
    save_path,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_detection = True,
    save_folder = './',
    load_detection = False,

    
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save precision recall curve of each label.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations
    detections_file = os.path.join(save_folder,'detections.txt')
    annotations_file = os.path.join(save_folder,'annotations.txt')

    if load_detection == True:
        with open(detections_file, "rb") as fp:  
            all_detections = pickle.load(fp)

        with open(annotations_file, "rb") as fp: 
            all_annotations = pickle.load(fp)

    else:
        all_detections     = _get_detections(generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
        all_annotations    = _get_annotations(generator)
        if save_detection == True:
            with open(detections_file, "wb") as fp:
                pickle.dump(all_detections, fp)

            with open(annotations_file, "wb") as fp:
                pickle.dump(all_annotations , fp)


    average_precisions = {}
    average_precisions_coco = {}
    for label in range(generator.num_classes()):
        average_precisions_coco[label] = []

    iou_values = np.arange(0.5, 1.00, 0.05).tolist()

    avg_map = []
    avg_map50 = []

    for label in range(generator.num_classes()):
        false_positives = []
        true_positives  = []
        for idx,iou_threshold1 in enumerate(iou_values):
            false_positives.append(np.zeros((0,)))
            true_positives.append( np.zeros((0,)))

        scores= np.zeros((0,))
        num_annotations = 0.0

        
        for i in range(len(generator)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []
            for idx, iou_threshold1 in enumerate(iou_values):
                detected_annotations.append([])

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    for idx,iou_threshold1 in enumerate(iou_values):
                        false_positives[idx] = np.append(false_positives[idx], 1)
                        true_positives[idx]  = np.append(true_positives[idx], 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                for idx,iou_threshold1 in enumerate(iou_values):
                    if max_overlap >= iou_threshold1 and assigned_annotation not in detected_annotations[idx]:
                        false_positives[idx] = np.append(false_positives[idx], 0)
                        true_positives[idx]  = np.append(true_positives[idx], 1)
                        detected_annotations[idx].append(assigned_annotation)
                    else:
                        false_positives[idx] = np.append(false_positives[idx], 1)
                        true_positives[idx]  = np.append(true_positives[idx], 0)


        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            average_precisions_coco[label].append(average_precision)
            continue

        # sort by score
        indices         = np.argsort(-scores)
        recall50=[]
        precision=[]
        for idx,iou_threshold1 in enumerate(iou_values):
            false_positives[idx] = false_positives[idx][indices]
            true_positives[idx]  = true_positives[idx][indices]

            # compute false positives and true positives
            false_positives[idx] = np.cumsum(false_positives[idx])
            true_positives[idx]  = np.cumsum(true_positives[idx])

            # compute recall and precision
            recall    = true_positives[idx] / num_annotations
            precision = true_positives[idx] / np.maximum(true_positives[idx] + false_positives[idx], np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = _compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations
            average_precisions_coco[label].append(average_precision)
            if iou_threshold1==0.5:
                recall50=recall
                precision50=precision


        print('\nmAP:')
        label_name = generator.label_to_name(label)
        print(len(average_precisions_coco[label]))
        print('mAP {}: {}'.format(label_name, mean(average_precisions_coco[label]) ))
        avg_map.append(mean(average_precisions_coco[label]))
        print('mAP50 {}: {}'.format(label_name, average_precisions_coco[label][0]))
        avg_map50.append(average_precisions_coco[label][0])
        print('mAP75 {}: {}'.format(label_name, average_precisions_coco[label][5]))
        print("Precision: ",precision[-1])
        print("Recall: ",recall[-1])
        
        if save_path!=None:
            plt.plot(recall50,precision50)
            # naming the x axis 
            plt.xlabel('Recall') 
            # naming the y axis 
            plt.ylabel('Precision') 

            # giving a title to my graph 
            plt.title('voxel '+label_name+'_Precision Recall curve') 
            plt.xlim([0, 0.8])
            # function to show the plot
            plt.savefig(save_path+'/'+label_name+'_precision_recall.jpg')
            plt.clf()
    print('mAP:{}'.format(mean(avg_map)))
    print('mAP50:{}'.format(mean(avg_map50)))


    return average_precisions_coco
