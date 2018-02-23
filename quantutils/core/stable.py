import subprocess
import levy as lStable
import numpy as np

def init(x):
    # Create the input file containing the data
    f = open('tempfile.txt', 'w')
    for i in x:
        f.write("%s\n" % i);
    f.close();

def fit(x, alpha=-1, beta=-1):
         
    # Calculate the stable parameters
    if alpha == 2 or (alpha == -1 and beta == -1):
        if alpha == 2:
            out = stable("7\ntempfile.txt\n5\n");
        else:
            out = stable("7\ntempfile.txt\n1\nY\n");
        
        # Extract the parameters from the output
        out = out[len(out) - 6].split();
        
        alpha = float(out[2]);
        beta =  float(out[3]);
        gamma = float(out[4]);
        delta = float(out[5]);       
    else:
        if alpha != -1 and beta == -1:
            out = lStable.fit_levy(x, alpha=alpha)
        elif alpha == -1 and beta != -1:
            out = lStable.fit_levy(x, beta=beta)
        elif alpha != -1 and beta != -1:
            out = lStable.fit_levy(x, alpha=alpha, beta=beta)
        
        alpha = out[0]
        beta =  out[1]
        gamma = out[3]
        delta = out[2]
    
    return [alpha, beta, gamma, delta];


    
def pdf(x, alpha, beta):
    
    return lStable.levy(x, alpha, beta);
    
def cdf(x, alpha, beta):
    
    return lStable.levy(x, alpha, beta, cdf=True);
    
def cdf2(alpha, beta, gamma, delta, steps, stepSize):
    insert1 = "%.7f\n%.7f\n0\n" % (alpha, alpha);
    insert2 = "%.7f\n%.7f\n0\n" % (beta, beta);
    insert3 = "%.7f\n%.7f\n" % (gamma, delta);
    x1 = delta - (int(steps / 2.0) * round(stepSize, 7))
    x2 = delta + (int(steps / 2.0) * round(stepSize, 7))
    inString = "2\n0\n%s%s%.7f\n%.7f\n%.7f\n%s1\n" % (insert1, insert2, x1, x2, stepSize, insert3)
    out = stable(inString);
    
    # Extract the results
    out = out[-(steps + 12):-12];
    out = [i.split() for i in out]
    x = [float(i[0]) for i in out]        
    y = [float(i[1]) for i in out]
    return (np.array(x),np.array(y));
    
def stable(stdin):  
    print "HELLO 2"
    # Call the STABLE.EXE program, input the data file
    p = subprocess.Popen(["C:\Users\cwilkin\Documents\stablec.exe"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate(input=stdin);
    out = out.split('\r\n');
    return out;    