import numpy as np

#Na parameters according to Alon:
# alpha m :
C1 = 0.1
C2 = 40
C3 = 40
C4 = 10
# beta m :
C5 = 4
C6 = 65
C7 = 18
# exponent of m :
C8 = 3
# alpha h :
c9 = 0.07
c10 = 65
c11 = 20
# beta h :
c12 = 35
c13 = 10

# Varmin Varmax
medium_varmin = [-1,    0, -1,   0, -1, -1,   0,  0, -1,   0,   0,   0,   0]
medium_varmax = [ 1,  100,  1, 100,  1,  1, 100, 10,  1, 100, 100, 100, 100]

# Gamma
medium_gamma = [[-0.01, 0.01], [1, 1], [-0.01, 0.01], [1, 1], [-0.01, 0.01], [-0.01, 0.01], [1, 1], [-0.01, 0.01],  [-0.01, 0.01],  [1, 1], [1, 1], [1, 1], [1, 1]]

# Sigma
medium_sigma = np.array([0.2, 4, 0.2, 4, 0.2, 0.2, 4, 0.2, 0.2, 4, 4, 4, 4])
