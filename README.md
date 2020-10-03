# Countour-Integral-for-Nonlinear-Eigenvalue-Problems 
(More detailed summary on the theory will follow)<br>
This is a Python implementation of the following paper https://arxiv.org/pdf/1003.1580.pdf<br>
Consider <img src="https://render.githubusercontent.com/render/math?math=T(z)v = 0">, 
where  <img src="https://render.githubusercontent.com/render/math?math=z\in \Gamma, v\in \mathbb{C}^m$">.
The problem is to find eigenpairs <img src="https://render.githubusercontent.com/render/math?math=(z, v)"> associated with the matrix-valued function  <img src="https://render.githubusercontent.com/render/math?math=z\mapsto T(z)\in \mathbb{C}^{m\times m}">  which is assumed to be holomorphic inside <img src="https://render.githubusercontent.com/render/math?math=\Gamma">.The problem is then reduced to evaluating
 <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{i2\pi}\int_\Gamma z^pT^{-1}(z)\hat{V}dz">

