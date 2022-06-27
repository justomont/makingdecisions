import re
import numpy as np
import csv
import ast
import colorsys
from copy import deepcopy
from operator import itemgetter


def anatomicREL(tag):
    region = ['Unknown',
               'Left-Cerebral-White-Matter',
               'Left-Cerebral-Cortex',
               'Left-Lateral-Ventricle',
               'Left-Inf-Lat-Vent',
               'Left-Cerebellum-White-Matter',
               'Left-Cerebellum-Cortex',
               'Left-Thalamus-Proper',
               'Left-Caudate',
               'Left-Putamen',
               'Left-Pallidum',
               '3rd-Ventricle',
               '4th-Ventricle',
               'Brain-Stem',
               'Left-Hippocampus',
               'Left-Amygdala',
               'CSF',
               'Left-Accumbens-area',
               'Left-VentralDC',
               'Left-vessel',
               'Left-choroid-plexus',
               'Right-Cerebral-White-Matter',
               'Right-Cerebral-Cortex',
               'Right-Lateral-Ventricle',
               'Right-Inf-Lat-Vent',
               'Right-Cerebellum-White-Matter',
               'Right-Cerebellum-Cortex',
               'Right-Thalamus-Proper',
               'Right-Caudate',
               'Right-Putamen',
               'Right-Pallidum',
               'Right-Hippocampus',
               'Right-Amygdala',
               'Right-Accumbens-area',
               'Right-VentralDC',
               'Right-vessel',
               'Right-choroid-plexus',
               '5th-Ventricle',
               'WM-hypointensities',
               'non-WM-hypointensities',
               'Optic-Chiasm',
               'CC Posterior',
               'CC Mid Posterior',
               'CC Central',
               'CC Mid Anterior',
               'CC Anterior']
    
    indx = [0,
            2,
            3,
            4,
            5,
            7,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            24,
            26,
            28,
            30,
            31,
            41,
            42,
            43,
            44,
            46,
            47,
            49,
            50,
            51,
            52,
            53,
            54,
            58,
            60,
            62,
            63,
            72,
            77,
            80,
            85,
            251,
            252,
            253,
            254,
            255] 
    
    anatomic = region[indx.index(tag)]
    return anatomic

def RAStoIJK(ras,volumeNode):
    # volumeNode = getNode('aseg')
    transformRasToVolumeRas = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas)
    point_VolumeRas = transformRasToVolumeRas.TransformPoint(ras[0:3])
    volumeRasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
    point_Ijk = [0, 0, 0, 1]
    volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas,1.0), point_Ijk)
    point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]
    return point_Ijk

def insideRES(nodeRAS):
    
    segmentationNode = getNode('Segmentation')
    
    try : 
        segmentLabelmapNode = getNode('res_map')
    except: 
        segmentLabelmapNode = slicer.vtkMRMLLabelMapVolumeNode()
        segmentLabelmapNode.SetName('res_map')
    
    # segmentLabelmapNode = slicer.vtkMRMLLabelMapVolumeNode()
    # segmentLabelmapNode.SetName('res_map')
    slicer.mrmlScene.AddNode(segmentLabelmapNode)
    segmentIDs = vtk.vtkStringArray()
    segmentationNode.GetSegmentation().GetSegmentIDs(segmentIDs)
    referenceNode = segmentationNode.GetNodeReference(slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole())
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, segmentIDs, segmentLabelmapNode, referenceNode)
    
    voxelArray = slicer.util.arrayFromVolume(segmentLabelmapNode)
    
    point_Ijk = RAStoIJK(nodeRAS,segmentLabelmapNode)
    
    try: 
        inside_val = voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]] # return 1 when the node is in the resection, 0 when not 
    except: 
        inside_val = 0
    
    return inside_val
    
def distance2res(nodeRAS):
    
    step = 0.05
    
    if insideRES(nodeRAS) == 1:
        distance = 0
    else:
        searching = True
        segmentationNode = getNode('Segmentation')
        segment = segmentationNode.GetSegmentation().GetNthSegment(0)
        pd = segment.GetRepresentation('Closed surface')
        com = vtk.vtkCenterOfMass()
        com.SetInputData(pd)
        com.Update()
        center = com.GetCenter()
        
        i = 1
        while searching:
            next_point_R = nodeRAS[0] + i*step*(center[0]-nodeRAS[0])  
            next_point_A = nodeRAS[1] + i*step*(center[1]-nodeRAS[1])
            next_point_S = nodeRAS[2] + i*step*(center[2]-nodeRAS[2])
            
            next_point = [next_point_R,next_point_A,next_point_S]
            i = i+1
            
            if insideRES(next_point) == 1: 
                searching = False
        
        distance = np.sqrt( (next_point[0]-nodeRAS[0])**2 + (next_point[1]-nodeRAS[1])**2 + (next_point[2]-nodeRAS[2])**2 )
        
    return distance

def distance2centre(onset,centre):
    
    indx = []
    
    for node in range(1,len(onset)): 
        dist2centre = np.sqrt( (centre[0]-onset[node][1])**2 + (centre[1]-onset[node][2])**2 + (centre[2]-onset[node][3])**2 )
        indx.append([onset[node][0], onset[node][1], onset[node][2], onset[node][3], float(dist2centre)])
        
    return indx

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return int((delta2 * (n - range1[0]) / delta1) + range2[0])

def showNodes(selected,single=True,markup_list='onset'):       
    'Def here'
    
    # Selected labels/nodes/channels
    labels = []
    done_labels = []
    pairs = []
    if single:
        labels = selected
    else:
        for pair in selected:
            labels.append(pair.split('-')[0])
            labels.append(pair.split('-')[1])
            pairs.append(pair.split('-'))

    # Select rulers
    annotationHierarchyNode = getNode('Ruler List') # rulers
    children = annotationHierarchyNode.GetNumberOfChildrenNodes() # number of rulers
    print(children)
    
    # Select the fiducials
    try:
        fidNode = getNode(markup_list)
    except:
        fidNode = slicer.vtkMRMLMarkupsFiducialNode()
        fidNode.SetName(markup_list)
        slicer.mrmlScene.AddNode(fidNode)
    
    # Sleect the volume
    volumeNode = getNode('aseg')
    voxelArray = slicer.util.arrayFromVolume(getNode('aseg'))
    
    # Initialize Onset list
    onset = [['Node','R','A','S','i','j','k','Anatomical Label','Anatomic Region']]
    
    for ruler_index in range(children):
        annotation = annotationHierarchyNode.GetNthChildNode(ruler_index).GetAssociatedNode() # one ruler
        if annotation != None:
            name = annotation.GetName() # name of the ruler e.g. 'A'
            for label in labels: # for each possible node
                if label in done_labels:
                    pass
                else:
                    # print(label)
                    num = int(re.search(r'\d+', label).group()) # extract the number
                    letter = ''.join([i for i in label if not i.isdigit()])
                    
                    print('  ')
                    print('node (read)', letter)
                    print(' name list (slicer)', name)
                    
                    if letter == name: # if the letter coincides
                        # print('\n name: '+name)
                        # print(letter)
                        if num == 1: # if the number is one we just have to select the start of the ruler
                            pstart = [0,0,0]
                            annotation.GetPosition1(pstart) # pstart now has the coordinates of the point 
                            fidNode.AddFiducialFromArray(pstart,letter+str(num))
                            ras = pstart
                            point_Ijk = RAStoIJK(ras,volumeNode)
                            onset.append([letter+str(num),ras[0],ras[1],ras[2],point_Ijk[0],point_Ijk[1],point_Ijk[2],voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]],anatomicREL(voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]])])
                            
                            done_labels.append(letter+str(num))
                        else:
                            pstart = [0,0,0]
                            annotation.GetPosition1(pstart) 
                            pend = [0,0,0]
                            annotation.GetPosition2(pend) 
                            measure = annotation.GetDistanceMeasurement()
                            location_x = pstart[0] + (num-1)*(3.5/measure)*(pend[0]-pstart[0])
                            location_y = pstart[1] + (num-1)*(3.5/measure)*(pend[1]-pstart[1])
                            location_z = pstart[2] + (num-1)*(3.5/measure)*(pend[2]-pstart[2])
                            location = [location_x,location_y,location_z]
                            fidNode.AddFiducialFromArray(location,letter+str(num))
                            ras = location
                            point_Ijk = RAStoIJK(ras,volumeNode)
                            onset.append([letter+str(num),ras[0],ras[1],ras[2],point_Ijk[0],point_Ijk[1],point_Ijk[2],voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]],anatomicREL(voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]])])
                            
                            done_labels.append(letter+str(num))
    return onset


def transformNodes(selected,transformation_name,single=True,markup_list='nodes'):       
    'Def here'
    
    # Selected labels/nodes/channels
    labels = []
    done_labels = []
    pairs = []
    if single:
        labels = selected
    else:
        for pair in selected:
            labels.append(pair.split('-')[0])
            labels.append(pair.split('-')[1])
            pairs.append(pair.split('-'))

    # Select rulers
    annotationHierarchyNode = getNode('Ruler List') # rulers
    children = annotationHierarchyNode.GetNumberOfChildrenNodes() # number of rulers
    
    # Select the fiducials
    try:
        fidNode = getNode(markup_list)
    except:
        fidNode = slicer.vtkMRMLMarkupsFiducialNode()
        fidNode.SetName(markup_list)
        slicer.mrmlScene.AddNode(fidNode)
    
    # Initialize Onset list
    onset = [['Node','R','A','S']]
    
    for ruler_index in range(children):
        annotation = annotationHierarchyNode.GetNthChildNode(ruler_index).GetAssociatedNode() # one ruler
        if annotation != None:
            name = annotation.GetName() # name of the ruler e.g. 'A'
            
            for label in labels: # for each possible node
                if label in done_labels:
                    pass
                else:
                    # print(label)
                    num = int(re.search(r'\d+', label).group()) # extract the number
                    letter = ''.join([i for i in label if not i.isdigit()])
                    print('node', letter)
                    print('name list', name)
                    
                    if letter == name: # if the letter coincides
                        print('\n name: '+name)
                        print(letter)
                        if num == 1: # if the number is one we just have to select the start of the ruler
                            pstart = [0,0,0]
                            annotation.GetPosition1(pstart) # pstart now has the coordinates of the point 
                            fidNode.AddFiducialFromArray(pstart,letter+str(num))
                            ras = pstart
                            mniToWorldTransformNode = getNode(transformation_name)  # replace this by the name of your actual transform
                            worldToMniTransform = vtk.vtkGeneralTransform()
                            mniToWorldTransformNode.GetTransformToWorld(worldToMniTransform)
                            mni=[0,0,0]
                            worldToMniTransform.TransformPoint(ras, mni)
                            onset.append([letter+str(num),mni[0],mni[1],mni[2]])
                            done_labels.append(letter+str(num))
                            
                        else:
                            pstart = [0,0,0]
                            annotation.GetPosition1(pstart) 
                            pend = [0,0,0]
                            annotation.GetPosition2(pend) 
                            measure = annotation.GetDistanceMeasurement()
                            location_x = pstart[0] + (num-1)*(3.5/measure)*(pend[0]-pstart[0])
                            location_y = pstart[1] + (num-1)*(3.5/measure)*(pend[1]-pstart[1])
                            location_z = pstart[2] + (num-1)*(3.5/measure)*(pend[2]-pstart[2])
                            location = [location_x,location_y,location_z]
                            fidNode.AddFiducialFromArray(location,letter+str(num))
                            ras = location
                            mniToWorldTransformNode = getNode(transformation_name)  # replace this by the name of your actual transform
                            worldToMniTransform = vtk.vtkGeneralTransform()
                            mniToWorldTransformNode.GetTransformToWorld(worldToMniTransform)
                            mni=[0,0,0]
                            worldToMniTransform.TransformPoint(ras, mni)
                            onset.append([letter+str(num),mni[0],mni[1],mni[2]])
                            done_labels.append(letter+str(num))
    return onset

def includeNodes(nodelist,patient_code):
    
    fidNode = slicer.vtkMRMLMarkupsFiducialNode()
    fidNode.SetName(patient_code)
    slicer.mrmlScene.AddNode(fidNode)
    
    for node in nodelist[1:]:
        ras = [node[1],node[2],node[3]]
        fidNode.AddFiducialFromArray(ras,node[0])
        
def paintMap(condition,freq,task,side=None):
            
    def paintNode(node):
        fidNode = slicer.vtkMRMLMarkupsFiducialNode()
        fidNode.SetName('NormMap')
        slicer.mrmlScene.AddNode(fidNode)
        
        ras = [float(node[0]),float(node[1]),float(node[2])]
        
        opacity = float(node[3])
        
        red = 1
        green = 1-float(node[3])
        blue = 1-float(node[3])
        
        
        fidNode.AddFiducialFromArray(ras,' ')
        fidNode.GetDisplayNode().SetGlyphType(7)
        fidNode.GetDisplayNode().SetGlyphScale(7)
        fidNode.GetDisplayNode().SetTextScale(0)
        fidNode.GetDisplayNode().SetOpacity(opacity)
        
        fidNode.GetDisplayNode().SetColor(red,green,blue)
        fidNode.GetDisplayNode().SetSelectedColor(red, green, blue)
    
    file = '/Volumes/GoogleDrive/Mi unidad/_BRAIN+COGNITION/0_TFM/Experiments/Analysis/brains/'+task.upper()+'/METABRAIN/NormMap/'+freq+'_'+condition+'.csv'
    
    with open(file) as f:
        reader = csv.reader(f)
        nodelist = list(reader)
    
    for node in nodelist:  
        
        if side:
            if side == 'R':
                if float(node[0]) > 0:
                    paintNode(node)
            if side == 'L':
                if float(node[0]) < 0:
                    paintNode(node)
        else:
            paintNode(node)
                    
