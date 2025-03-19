#%%
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt


#%%
plt.hist(pm.draw(pm.Gamma.dist(2, 4), draws=100))

#%%
ls = 2
tau = 1.
cov = tau**2 * pm.gp.cov.Matern52(1, ls)

X = np.linspace(-4, 4, 200)[:, None]
K = cov(X).eval()
gp_prior = pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=len(K))
gp_samples = pm.draw(gp_prior, draws=20, random_seed=42),

plt.plot(
    X,
    gp_samples[0].T,
    alpha=0.5,
    color="#AAAAAA",
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X")
# %%
