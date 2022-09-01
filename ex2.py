#~ ======================================================================
#~ CREATION JEU DE DONNEES
import numpy as np
from numpy.random import rand, seed
from pandas import concat, DataFrame

def foo(ncols):
	row = { f'X{i}': rand() for i in range(ncols) }
	row.update({ 'Y': 13 * row['X0'] + 7 * row[f'X{ncols-1}'] + 31 + rand() })
	return row

seed(0) # Je force l'initialisation du générateur pseudo-aléatoire
ncols = 13
df = DataFrame(data=[foo(ncols) for _ in range(125)])

#~ df = df.head(14)
nomsVariables = df.columns[:-1].tolist()

print(f'{len(df)} individus, {len(df.columns)-1} variables {nomsVariables}')

#~ ======================================================================
#~ CORRELATION
#~ method='kendall', 'spearman'
corr_df = df.corr(method='pearson')

print('\nCoefficients de correlation\n', corr_df)

x, y = df.X12, df.Y
maCorr = np.dot(x-x.mean(), y-y.mean()) / ((len(x)-1) * x.std() * y.std())
print(f'Correl : {maCorr:.6f}')

#~ -----
import matplotlib.pyplot as plt

#~ plt.figure(figsize=(8.26, 8.26))
plt.matshow(corr_df)
plt.show()

#~ -----
import seaborn as sns
plt.figure(figsize=(8.26, 8.26))
sns.heatmap(corr_df, annot=True)
plt.show()

#~ ======================================================================
X = df.drop(columns=['Y'])
y = df.Y

#~ ======================================================================
#~ Use StandardScaler to help you standardize the dataset's features onto unit scale 
#~ (mean = 0 and variance = 1)
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)

#~ ======================================================================
#~ Principal Component Analysis (PCA) is a linear dimensionality reduction technique that can be
#~ utilized for extracting information from a high-dimensional space by projecting it into a
#~ lower-dimensional sub-space.
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

#~ The new components are just the two main dimensions of variation
principalDf = DataFrame(data = principalComponents,
	columns = ['pc1', 'pc2'])

finalDf = concat([ principalDf, df[['Y']] ], axis=1)

#~ *** What are the limitations of PCA?
	#~ * PCA is not scale invariant. check: we need to scale our data first.
	#~ * The directions with largest variance are assumed to be of the most interest
	#~ * Only considers orthogonal transformations (rotations) of the original variables
	#~ * PCA is only based on the mean vector and covariance matrix. Some distributions (multivariate normal) are characterized by this, but some are not.
	#~ * If the variables are correlated, PCA can achieve dimension reduction. If not, PCA just orders them according to their variances.
#~ ======================================================================
#~ CLUSTERING
from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(2.8 * 8.26, 1.2 * 8.26))
Z = linkage(finalDf, method='ward', metric='euclidean')
d = dendrogram(Z, orientation='top', show_contracted=True, no_plot=False)
nclusters = len(list(set(d['color_list']))) - 1
plt.title(f'Classification par dendrogramme (nclusters={nclusters})', fontsize=24)
plt.grid(False)
plt.show()

#~ ======================================================================
#~ CLUSTERING
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=nclusters, init='k-means++', n_init=10, max_iter=300)
kmeans = kmeans.fit(finalDf)
finalDf['cluster'] = kmeans.predict(finalDf)

#~ ======================================================================
nrows, ncols = 1, 1
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*8.26, nrows*8.26))

ax.set_xlabel('Principal Component 1', fontsize=20)
ax.set_ylabel('Principal Component 2', fontsize=20)
ax.set_title('2 component PCA', fontsize = 24)

for c in range(nclusters):
	_df = finalDf[ finalDf.cluster == c ]
	ax.scatter(_df.pc1, _df.pc2, label=f'Cluster {c}')

ax.legend(loc='upper right', prop={'size': 20})
ax.grid()
plt.show()
plt.close(fig)

#~ ======================================================================
from scipy.stats import chi2_contingency 

stat, p, dof, expected = chi2_contingency(df) 

alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
	print('Dependant (rejet de H0)')
else: 
	print('Independant (H0 est vraie)')
