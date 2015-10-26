import sys
import numpy as np
sys.path.append("/usr/local/lib/python2.7/site-packages/")

from BackboneDBN.DBN import FB5DBN, TorusDBN


f = open("animation2.txt","r")
lines = f.readlines()

for i in xrange(len(lines)/3):
    #seq = lines[i*3].strip()
    seq = lines[0].strip()
    phi = np.fromstring(lines[i*3+1].strip().strip("[]"), sep=",")
    psi = np.fromstring(lines[i*3+2].strip().strip("[]"), sep=",")
    
    ang = []
    for (a,b) in zip(phi, psi):
        ang.append([a,b])
    
    #print seq
    #print ang
    dbn = TorusDBN(start_state="UNIFORM",transition_state="UNIFORM")
    dbn.init(angles_seq=ang, aa_seq=seq)
    structure = dbn.get_structure()

    dbn.save_structure("sample%d.pdb" % i, structure)
