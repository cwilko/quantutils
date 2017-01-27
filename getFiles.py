import subprocess
import xml.etree.ElementTree as ET


#out = subprocess.check_output(["curl", "-k", "-c", '\"E:/cookies.txt\"', '\"https://jazz609.hursley.ibm.com:9443/jazz/authenticated/identity\"'], stderr=subprocess.STDOUT, shell=True)
out = subprocess.check_output("curl -k -c \"E:\cookies.txt\" \"https://jazz609.hursley.ibm.com:9443/jazz/authenticated/identity\"", stderr=subprocess.STDOUT, shell=True)


#proc2 = subprocess.Popen(["curl", "-k", "-c", "\"E:\\cookies.txt\"", "-b", "\"E:\\cookies.txt\"", "-L", "-d", "j_username=chris.wilkinson@uk.ibm.com", "-d", "j_password=iux5s9ie", "\"https://jazz609.hursley.ibm.com:9443/jazz/authenticated/j_security_check\""], stdout=subprocess.PIPE)
out = subprocess.check_output("curl -k -L -b \"E:\cookies.txt\" -c \"E:\cookies.txt\" -d j_username=chris.wilkinson@uk.ibm.com -d j_password=iux5s9ie \"https://jazz609.hursley.ibm.com:9443/jazz/authenticated/j_security_check\"", stderr=subprocess.STDOUT, shell=True)


prevDefect = -1
defect = 0
inc = 0

while defect != prevDefect :
    
    inc = inc + 1
    curl = "curl -k -c \"E:\cookies.txt\" -b \"E:\cookies.txt\" -L \"https://jazz609.hursley.ibm.com:9443/jazz/rpt/repository/workitem?fields=workitem/workItem\\[projectArea/name=%27APIM%27%20and%20id>" + str(defect) + "\\]/(id|itemHistory/(stateId|predecessor|modified|state/id|target/name|resolution/id))\" > \"E:/file" + str(inc) + ".xml\""

    print curl
    out = subprocess.check_output(curl, stderr=subprocess.STDOUT, shell=True)
        
    tree = ET.parse("E:/file" + str(inc) + ".xml")
    root = tree.getroot()
    
    prevDefect = defect
    
    print ("PARSED");
    
    for workitem in root.findall('workItem'):
        defect = int(workitem.find('id').text)
        
    print ("Retrieved page " + str(inc) + " up to defect " + str(defect))
