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

"""
ISDF-THC form:
Vpqrs = Xpm Xrm Vmn Xqn Xsn 

CP4 form:
Vpqrs = Ypl Yql Yrl Ysl

FIXME: this is only correct, if fully symmetric
"""

def get_AB(A,B):
    AB = np.einsum('pm,pl->ml',A,B)
    return AB

def get_XYVXYsq(Vcore,XY):
    XYsq = np.einsum('ml,ml->ml',XY,XY)
    VXYsq = np.einsum('mn,nl->ml',Vcore,XYsq)
    XYVXYsq = np.einsum('ml,ml->ml',XY,VXYsq)
    return  XYVXYsq

def get_RHS(XYVXYsq,X):
    RHS = np.einsum('pm,ml->pl',X,XYVXYsq)
    return RHS

def get_Y6(YY):
    Y4 = np.einsum('Ll,Ll->Ll',YY,YY)
    Y6 = np.einsum('Ll,Ll->Ll',Y4,YY)
    return Y6

def get_Y(RHS,Y6,Y_old):
    e,v = np.linalg.eigh(Y6)
    Rv = np.einsum('pL,Ls->ps',RHS,v)
    Yv_old = np.einsum('pl,ls->ps',Y_old,v)
    nmo = Y_old.shape[0]
    S = np.tile(e,(nmo,1))
    Sreg = np.where(np.abs(S) > 1e-10, S, 1.)
    Sinv = 1. / Sreg
    Yv_new = np.where(Sinv > 1e-10, Rv * Sinv, Yv_old)
    Y_new = np.einsum('ps,ls->pl',Yv_new,v)
    return Y_new

def get_fid(XY,X,YY,Vcore,XYVXYsq,Y6):
    XX = get_AB(X,X)
    XXsq = np.einsum('ma,ma->ma',XX,XX)
    mixed = 2. * np.einsum('ml,ml',XY,XYVXYsq)
    cp = np.einsum('lL,lL',YY,Y6)
    thc_ = np.einsum('mn,nb->mb',Vcore,XXsq)
    thc = np.einsum('mb,bm',thc_,thc_)
    return thc - mixed + cp 

def get_CP4(X,Vcore,cp_rank_factor=2,max_iter=100,threshold=1e-4):
    ncore = Vcore.shape[0]
    ncp = ncore*cp_rank_factor
    nmo = X.shape[0]
    # get init guess
    Y = np.random.rand(nmo,ncp)
    # Y = np.zeros((nmo,ncp))

    # optimize Y
    for cp_iter in range(max_iter):
        XY = get_AB(X,Y)
        YY = get_AB(Y,Y)
        XYVXYsq = get_XYVXYsq(Vcore,XY)
        RHS = get_RHS(XYVXYsq,X)
        Y6 = get_Y6(YY)
        Y_new = get_Y(RHS,Y6,Y)
        norm_diff = np.linalg.norm(Y_new - Y)/ np.linalg.norm(Y)
        fid = get_fid(XY,X,YY,Vcore,XYVXYsq,Y6)
        print(f'Iteration {cp_iter}: norm diff {norm_diff:.6e}, fid {fid:.6e}')  
        if norm_diff < threshold:
            print(f'Converged after {cp_iter} iterations with norm diff {norm_diff:.6e} and fid {fid:.6e}')
            break
        elif cp_iter == max_iter - 1:
            print(f'Warning: did not converge after {max_iter} iterations, norm diff {norm_diff:.6e} and fid {fid:.6e}')
        
        Y = Y_new

    return Y

# test with random
#########
nmo = 10
nocc = 5
nvir = nmo - nocc

ncore_fac = 3
ncore = nmo * ncore_fac

X = np.random.rand(nmo,ncore)
Vcore_ = np.random.rand(ncore,ncore)
Vcore = 0.5 * (Vcore_ + Vcore_.T)  # make symmetric

Y = get_CP4(X,Vcore,cp_rank_factor=10,max_iter=100,threshold=1e-4)


##########

