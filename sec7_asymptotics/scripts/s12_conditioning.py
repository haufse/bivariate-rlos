"""s12: conditioning of Sigma on T and robustness of the integrated ratios V_r/V_1 (2026-11-18).

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-9 nice -n 19 \
      python s12_conditioning.py > ../outputs/s12_output.txt
Needs ../outputs/sigma_grid_theta*.npz (s04).  Seed: numpy default_rng(1) for the perturbation test.
For theta in {2,3,4,6} and T = [0.15,0.85], [0.01,0.99]: max condition numbers of Sigma_2, Sigma_3 on T, and
V_2/V_1, V_3/V_1 computed (i) with np.linalg.solve (as in s05/s08), (ii) with eigenvalues below 1e-12 and
1e-9 times the largest dropped, (iii) after a symmetric random perturbation of size 1e-14 (the quadrature
accuracy of the entries).  Agreement of (i), (ii, 1e-12) and (iii) shows the ill-conditioning near the
endpoints does not affect the reported ratios.  (ii, 1e-9) is too aggressive: it discards informative
directions; ratios above one there are an artefact of the truncation.
"""
import sys; sys.path.insert(0,'.')
import numpy as np, kquad, rlos_lib as rl
rng=np.random.default_rng(1)
IDX={2:[0,1,3,4],3:list(range(9))}
def sopt(S,d,tol=None):
    if tol is None: return 1/(d@np.linalg.solve(S,d))
    ev,U=np.linalg.eigh(S); keep=ev>tol*ev.max()
    y=U[:,keep].T@d
    return 1/np.sum(y**2/ev[keep])
for th in [2.0,3.0,4.0,6.0]:
    D=np.load(f'../outputs/sigma_grid_theta{th:g}.npz')
    for de in (0.15,0.01):
        t,w,S,_=kquad.half_nodes(D,de)
        c2=max(np.linalg.cond(s_[np.ix_(IDX[2],IDX[2])]) for s_ in S); c3=max(np.linalg.cond(s_) for s_ in S)
        print(f'theta={th:g} T=[{de},{1-de}]: max cond Sigma_2 {c2:.1e}, Sigma_3 {c3:.1e}')
        res={}
        for tag,tol,pert in (('solve',None,0),('tol1e-12',1e-12,0),('tol1e-9',1e-9,0),('pert1e-14',None,1e-14)):
            v1=v2=v3=0
            for tt,ww,Sg in zip(t,w,S):
                A=float(rl.A_log(tt,th)); d=rl.d_vec(A,tt,3)
                if pert:
                    E=rng.normal(size=(9,9)); E=(E+E.T)/2*pert; Sg=Sg+E
                v1+=ww*Sg[0,0]/d[0]**2
                v2+=ww*sopt(Sg[np.ix_(IDX[2],IDX[2])],d[IDX[2]],tol)
                v3+=ww*sopt(Sg,d,tol)
            res[tag]=(v2/v1,v3/v1)
        print(th,de,' '.join(f'{k}: {a:.5f}/{b:.5f}' for k,(a,b) in res.items()))
