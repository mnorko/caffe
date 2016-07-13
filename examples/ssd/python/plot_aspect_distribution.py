from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-f', '--trainval_file', type=str, required=True)


# function to read the ground truth bounding box information
import xmltodict

def parse_gt(file_path):
    f = open(file_path, 'r')
    #f = open('/home/saurabhkumar/ssd/caffe/data/VOC0712/test.txt', 'r')
    lines = f.read().split("\n")

    widths = []
    heights = []
    areas = []
    bbox_coordinates = []

    count = 0
    
    for i in xrange(len(lines) - 1):
        line = lines[i]
        image_annotation = line.split(" ")
        annotation_file = image_annotation[1]
        #annotation_file = '/home/saurabhkumar/ssd/caffe/VOCdevkit/' + annotation_file
        
        with open(annotation_file) as fd:
            doc = xmltodict.parse(fd.read())

        image_width = float(doc['annotation']['size']['width'])
        width_ratio = float(300) / float(image_width)

        image_height = float(doc['annotation']['size']['height'])
        height_ratio = float(300) / float(image_height)

        image_area = image_width * image_height
        #print image_area
        area_ratio = float(300*300) / float(image_area)

        try:
            objs = doc['annotation']['object']

            if type(objs) == list:
                for obj in objs:
                    #print obj['bndbox']
                    bnd_boxes = obj['bndbox']
                    x_min = int(bnd_boxes['xmin'])
                    x_max = int(bnd_boxes['xmax'])
                    y_min = int(bnd_boxes['ymin'])
                    y_max = int(bnd_boxes['ymax'])

                    width = int(x_max) - int(x_min)
                    height = int(y_max) - int(y_min)
                    area = width * height

                    width = width_ratio * float(width)
                    height = height_ratio * float(height)
                    area = area_ratio * float(area)

                    x_min = x_min * width_ratio
                    x_max = x_max * width_ratio
                    y_min = y_min * height_ratio
                    y_max = y_max * height_ratio

                    #Squash coordinates between 0 and 1
                    x_min /= float(300)
                    x_max /= float(300)
                    y_min /= float(300)
                    y_max /= float(300)

                    widths.append(width)
                    heights.append(height)
                    areas.append(area)
                    bbox_coordinates.append((x_min, y_min, x_max, y_max))

            else:
                bnd_boxes = doc['annotation']['object']['bndbox']
                x_min = int(bnd_boxes['xmin'])
                x_max = int(bnd_boxes['xmax'])
                y_min = int(bnd_boxes['ymin'])
                y_max = int(bnd_boxes['ymax'])


                width = int(x_max) - int(x_min)
                height = int(y_max) - int(y_min)
                area = width * height

                width = width_ratio * float(width)
                height = height_ratio * float(height)
                area = area_ratio * float(area)

                x_min = x_min * width_ratio
                x_max = x_max * width_ratio
                y_min = y_min * height_ratio
                y_max = y_max * height_ratio

                #Squash coordinates between 0 and 1
                x_min /= float(300)
                x_max /= float(300)
                y_min /= float(300)
                y_max /= float(300)

                widths.append(width)
                heights.append(height)
                areas.append(area)
                bbox_coordinates.append((x_min, y_min, x_max, y_max))
        

        except:
            #print type(doc['annotation']['object'])
            #print ""
            count += 1
            #print count

    #print count
    #print widths
    #print areas
    #print len(widths)
    #print len(areas)
    
    return bbox_coordinates


def jaccard_overlap(bbox1, bbox2):
    # Return jaccard overlap of the two input bounding boxes
    
    (x_min1, y_min1, x_max1, y_max1) = bbox1
    (x_min2, y_min2, x_max2, y_max2) = bbox2
    
    intersection = max(0, min(x_max1, x_max2) - max(x_min1, x_min2)) * max(0, min(y_max1, y_max2) - max(y_min1, y_min2))
    union = (x_max1 - x_min1) * (y_max1 - y_min1) + (x_max2 - x_min2) * (y_max2 - y_min2) - intersection
     
    overlap = float(intersection) / float(union)
    
    return overlap

import math

'''
    For a given gt_bbox, return max jaccard overlap score
    for any prior box for this feature map and the number of
    overlap scores greater than threshold parameter
'''
def gt_priorbox_match(image_size, featmap, aspect_ratio_arr, min_size, max_size, gt, thresh=0.5):
    #print min_size
    #print max_size
    # example image_size: [300,300]
    # example feat_map_size: [38,38]
    # aspect_ratio: [2], min_size: 30, max_size -1
    
    aspect_ratios = []
    for aspect_ratio in aspect_ratio_arr:
        aspect_ratios.append(aspect_ratio)
        aspect_ratios.append(1 / float(aspect_ratio))
    
    
    layer_width = featmap[0]
    layer_height = featmap[1]
    img_width = image_size[0]
    img_height = image_size[1]
    
    step_x = float(img_width) / float(layer_width)
    step_y = float(img_height) / float(layer_height)
    
    top_data = []
    
    for h in xrange(layer_height):
        for w in xrange(layer_width):
            center_x = float((w + 0.5) * step_x)
            center_y = float((h + 0.5) * step_y)
            
            # first prior: aspect ratio = 1, size = min_size
            box_width = min_size
            box_height = min_size
            # xmin
            top_data.append((center_x - box_width / 2.) / img_width)
            # ymin
            top_data.append((center_y - box_height / 2.) / img_height)
            # xmax
            top_data.append((center_x + box_width / 2.) / img_width)
            # ymax
            top_data.append((center_y + box_height / 2.) / img_height)
        
            
            if max_size > 0:
                # second prior: aspect ratio = 1, size = sqrt(min_size * max_size)
                box_width = float(math.sqrt(min_size * max_size))
                box_height = float(math.sqrt(min_size * max_size))
                # xmin
                top_data.append((center_x - box_width / 2.) / img_width)
                # ymin
                top_data.append((center_y - box_height / 2.) / img_height)
                # xmax
                top_data.append((center_x + box_width / 2.) / img_width)
                # ymax
                top_data.append((center_y + box_height / 2.) / img_height)
            
            #rest of priors
            for r in xrange(len(aspect_ratios)):
                ar = float(aspect_ratios[r])
                
                box_width = min_size * float(math.sqrt(ar))
                box_height = min_size / float(math.sqrt(ar))
                # xmin
                top_data.append((center_x - box_width / 2.) / img_width)
                # ymin
                top_data.append((center_y - box_height / 2.) / img_height)
                # xmax
                top_data.append((center_x + box_width / 2.) / img_width)
                # ymax
                top_data.append((center_y + box_height / 2.) / img_height)
        
    # clip the prior's coordinates so that they are within [0, 1]
    for d in xrange(len(top_data)):
        top_data[d] = max(top_data[d], 0)
        top_data[d] = min(top_data[d], 1)

    overlap_over_threshold_count = 0
    best_overlap = 0

    for i in xrange(0, len(top_data), 4):
        prior_box = (top_data[i], top_data[i + 1], top_data[i + 2], top_data[i + 3])
        overlap = jaccard_overlap(prior_box, gt)
        if best_overlap < overlap:
            best_overlap = overlap
        if overlap > thresh:
            overlap_over_threshold_count += 1

    #print top_data
    #print best_overlap
    return [best_overlap, overlap_over_threshold_count]

import numpy as np

'''
    Compute histogram of feature maps and gtboxes
'''
def compute_histogram():
    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'pool6']
    min_dim = 300
    
    feat_maps = [[38,38],[19,19],[10,10],[5,5],[3,3],[1,1]]
    #aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    
    min_ratio = 20
    #min_ratio = 10
    max_ratio = 95
    #max_ratio = 80
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 10 / 100.] + min_sizes
    #min_sizes = [min_dim * 5 / 100.] + min_sizes 
    max_sizes = [-1] + max_sizes
    
    print "Prior box sizes: " + str(min_sizes)
    
    image_size = [300,300]
    gt_boxes = parse_gt()
    
    hist = [0 for i in range(len(feat_maps))]
    best_matches_greater_thresh = [0 for i in range(len(feat_maps))]
    best_matches_smaller_thresh = [0 for i in range(len(feat_maps))]

    for gt_index in xrange(len(gt_boxes)):
        gt = gt_boxes[gt_index]
        layer = None
        best_overlap = 0.0
        for i in range(len(feat_maps)):
            [layer_best_overlap, overlap_over_threshold_count] = gt_priorbox_match(image_size, feat_maps[i], aspect_ratios[i], min_sizes[i], max_sizes[i], gt)
            # Match each ground truth box to the default box with the best jaccard overlap.
            if best_overlap < layer_best_overlap:
                layer = i #keep track of layer for which overlap is best
                best_overlap = layer_best_overlap 

            # Match each default box to any ground truth with jaccard overlap higher than 0.5
            #COMMENTED OUT!!!
            hist[i] += overlap_over_threshold_count


        # increment matched layer count
        # Prevent adding a redundant count to hist[layer] if it was already added when adding count for overlap over threshold
        if layer is not None and best_overlap <= 0.5: 
            hist[layer] += 1
            best_matches_smaller_thresh[layer] += 1
        
        #ADDED
        if layer is not None and best_overlap > 0.5:
            best_matches_greater_thresh[layer] += 1

        
    #Make the plots!
    print "Prior box sizes: " + str(min_sizes)
    print "Number of ground truth bounding boxes: " + str(len(gt_boxes))
    print "Aspect ratios: " + str(aspect_ratios)
    
    print "Total number of overall matches: " + str(sum(hist))
    print "Total number of best matches with overlap <= 0.5: " + str(sum(best_matches_smaller_thresh))
    
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    import numpy as np

    ind = np.arange(6)

    fig = plt.figure()
    fig.set_size_inches(17, 8)
    ax = fig.add_subplot(111)

    ## necessary variables
    width = .2  # the width of the bars
    
    # normalize the bars
    
    
    hist_normalized = [float(x) * 100 / sum(hist) for x in hist]
    best_matches_greater_thresh_normalized = [float(x) * 100 / sum(best_matches_greater_thresh) 
            for x in best_matches_greater_thresh]
    best_matches_smaller_thresh_normalized = [float(x) * 100 / sum(best_matches_smaller_thresh) 
            for x in best_matches_smaller_thresh]
    

    ## the bars
    rects1 = ax.bar(ind, hist_normalized, width,
                    color='red')
    
    rects2 = ax.bar(ind + width, best_matches_smaller_thresh_normalized, width, 
                    color='blue')
    
    rects3 = ax.bar(ind + 2*width, best_matches_greater_thresh_normalized, width, 
                    color='green')
    
    
    # axes and labels
    ax.set_xlim(0,max(ind)+width*3)
    ax.set_ylim(0, 101)
    ax.set_ylabel('Percentage of Matches')
    ax.set_xlabel('Layer')
    ax.set_title('Distribution of Bounding Box Matches')
    ax.set_xticks(ind + 1.5*width)
    ax.set_xticklabels(('conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'pool11'))
    ax.legend((rects1[0], rects2[0], rects3[0]), 
              ('Overall Matches', 'Best Matches <= 0.5', 'Best Matches > 0.5'))

    #plt.show()

def main(args):
	file_path = args.trainval_file
	compute_histogram(file_path)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
