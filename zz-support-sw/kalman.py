import numpy as np

# intial parameters
n_iter = 50
sz = (n_iter,) # size of array
x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)

Q = 1e-5 # process variance

# allocate space for arrays
xhat=np.zeros(sz)      # a posteri estimate of x
P=np.zeros(sz)         # a posteri error estimate
xhatminus=np.zeros(sz) # a priori estimate of x
Pminus=np.zeros(sz)    # a priori error estimate
K=np.zeros(sz)         # gain or blending factor

R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1,n_iter):
	# time update
	xhatminus[k] = xhat[k-1]
	Pminus[k] = P[k-1]+Q

	# measurement update
	K[k] = Pminus[k]/( Pminus[k]+R )
	xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
	P[k] = (1-K[k])*Pminus[k]