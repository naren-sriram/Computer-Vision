import xml.etree.ElementTree as ET
import numpy as np
tree = ET.parse('AP35X9339-2.xml') #Give any .xml file as input here
root = tree.getroot()

#Read values from xml file
def xml_read(root=[]):
    detections=[]
    for child in root:
        if child.tag=="object":
            for name in child:
                if name.tag == "name":
                    number_name = name.text
                if name.tag == "bndbox":
                    bndbox_val = [] 
                    for values in name:
                        bndbox_val.append(int(values.text))
                try:
                    if bndbox_val not in detections:
                        detections.append(bndbox_val)
                except:
                    pass
    return detections

#Re-arrange the values to mimic the problem statement as xml values are in order
def rearrange(detections):
    detect = []
    for i in range(len(detections)):
        if (i%2)==0:
            detect.append(detections[i])
    for i in range(len(detections)):
        if (i%2)!=0:
            detect.append(detections[i])
    detections = detect
    return detections

#Identify mean of ymins
def mean_of_ymins(detections):
    list_of_all_ymins = []
    for bndbox in detections:
        list_of_all_ymins.append(bndbox[1])
    mean_y = sum(list_of_all_ymins)/len(list_of_all_ymins)
    return mean_y
    
#Cluster the bounding boxes based on:
#if ymin < ymean, put into row1, otherwise row2
#Sort each row based on ascending order of xmin and return the 2 rows in order.
def cluster_sort_bndbox(detections, mean_y=0, detections_row2=[], detections_row1=[]):
    for bndbox in detections:
        if bndbox[1] < mean_y:
            detections_row1.append(bndbox)
    for bndbox in detections:
        if bndbox[1] > mean_y:
            detections_row2.append(bndbox)
    detections_row1 = np.array(detections_row1)
    detections_row2 = np.array(detections_row2)
    print("Number plate in order:", detections_row1[detections_row1[:,0].argsort()])
    print(detections_row2[detections_row2[:,0].argsort()])
    return list(zip(detections_row1,detections_row2))
    

detections = xml_read(root)
print("Values read from XML: ", detections)
print("Re-arranged: ", rearrange(detections))
#If 'detections' is given as an array of bounding boxes, actual code starts from here
mean_y = mean_of_ymins(detections)
print("Mean of all ymins: ",mean_y)
cluster_sort_bndbox(detections, mean_y, detections_row2 = [], detections_row1 = [])


