import xml.etree.ElementTree as ET
from datetime import *

f = open("E:/myfile.csv",'w')


keyDate = datetime.strptime("2014-07-18", "%Y-%m-%d").date()

for j in range(1,20):
    
    tree = ET.parse("E:/file" + str(j) + ".xml")
    root = tree.getroot()


    for workitem in root.findall('workItem'):
        wId = workitem.find('id').text       
        modMap = dict()
        preMap = dict()
        targetMap = dict() 
        itemStateMap = dict()
        states = []
        resMap = dict()
        orderedStates = list()
        orderedItemStates =list()
        for itemHistory in workitem.findall('itemHistory'):
            modified = (itemHistory.find('modified').text)[:10]
            stateId = itemHistory.find('stateId').text
            previous = itemHistory.find('predecessor').text
            itemState = itemHistory.find('state').find('id').text
            target = itemHistory.find('target').find('name')
            iteration = "Unassigned";
            if target != None:
                iteration = target.text
            resolution = itemHistory.find('resolution')
            res = "Unnasigned"
            if resolution != None:
                resolution = resolution.find('id')
                if resolution != None:
                    if resolution.text != None:
                        res = resolution.text                   
            states.append(stateId)
            modMap[stateId] = datetime.strptime(modified, "%Y-%m-%d")
            preMap[previous] = stateId
            itemStateMap[stateId] = itemState
            targetMap[stateId] = iteration
            resMap[stateId] = res
        orderedStates.append(preMap[None]);
        for i in range(1,len(preMap)):
            orderedStates.append(preMap[orderedStates[i-1]])
            if modMap[orderedStates[i]].date() >= keyDate:
                print (wId + "," + orderedStates[i-1] + "," + itemStateMap[orderedStates[i-1]] + "," + resMap[orderedStates[i-1]] + "," + str(modMap[orderedStates[i-1]]))
                output = (wId + "," + itemStateMap[orderedStates[i-1]] + "," + resMap[orderedStates[i-1]] + "," + targetMap[orderedStates[i-1]])
                f.write(output + "\n")
                break;
        
            
           
f.close(); 
            
                                    
        