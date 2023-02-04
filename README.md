# boundHHoffman
Suppose $A\in \mathbb{R}^{m\times n}\setminus \{0\}$ and $P:=\{x:Ax\le 0\}$.

This is a Python implementation of the procedure described in the paper "An easily computable upper bound on the Hoffman constant for homogeneous inequality systems" to compute an upper bound on the following *homogeneous Hoffman constant:*

$$
H_0(A) := \sup_{u\in \mathbb{R}^n \setminus P} \frac{\text{dist}(u,P)}{\text{dist}(Au, \mathbb{R}^m_-)}.$$  

The main procedure is implemented in the file "boundHHoffman.py".

The notebook "BoundHHoffman.ipynb" includes some illustrative examples.
