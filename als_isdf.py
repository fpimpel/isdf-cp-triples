import numpy as np

""" 
Script to approximate the ISDF-THC coulomb integral by a CP4 form 
"""
# open questions
"""  
Better to do for pqrs space or only for a slice? eg pqri
full better symmetry, slice smaller

How big CP4 internal rank?

If fully symmetric, what is best way to optimize?
"""

# test with random
#########
nmo = 50
nocc = 5
nvir = nmo - nocc

ncore_fac = 3
ncore = nmo * ncore_fac

X = np.random.rand((nmo,ncore))
Vcore = np.random.rand((nmo,nmo))
##########

"""
ISDF-THC form:
Vpqrs = Xpm Xrm Vmn Xqn Xsn 

CP4 form:
Vpqrs = Ypl Yql Yrl Ysl

FIXME: this is only correct, if fully symmetric
"""

def get_XY(X,Y):
    XY = np.einsum('pm,pl->ml',X,Y)
    return XY

def get_VXY(Vcore,XY):
    XYsq = XY * XY
    VXYsq = np.einsum('mn,nl->ml',Vcore,XYsq)
    return VXYsq

#def get_norm_diff(X)

def get_CP4(X,Vcore,cp_rank_factor=2,max_iter=100,threshold=1e-6):
    ncore = Vcore.shape[0]
    ncp = ncore*cp_rank_factor
    nmo = X.shape[0]
    # get init guess
    Y = np.random.rand((nmo,ncp))

    # optimize Y
    for cp_iter in range(max_iter):
        XY = get_XY(X,Y)
        

    return Y

