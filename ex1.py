#~ ======================================================================
#~ CHARGEMENT JEU DE DONNEES
from numpy import linspace
from numpy.random import rand
from pandas import concat, DataFrame, read_excel

df = read_excel('./dataset_1.xls')

#~ ALTERNATIVE DE CREATION DU pandas.DataFrame
'''
def foo():
	x1, x2, x3 = rand(3)
	return { 'X1': x1, 'X2': x2, 'X3': x3, 'Y': 13 * x2 + 7 + rand() }
df = DataFrame(data=[foo() for _ in range(125)])
'''
df = df.head(30)
nomsVariables = df.columns[:-1].tolist()

print(f'{len(df)} individus, {len(df.columns)-1} variables {nomsVariables}')

#~ ======================================================================
print('\nCoefficients de correlation\n', df.corr(method='pearson'))

#~ ======================================================================
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

for nomVariable in nomsVariables:
	X = df[nomVariable]
	y = df['Y']

	X = X.to_numpy().reshape(-1,1)

	#~ model = make_pipeline(StandardScaler(), LinearRegression())
	model = LinearRegression()
	model.fit(X, y)
	score = model.score(X, y)

	#~ lr = model['linearregression']
	lr = model

	p, q = lr.coef_[0], lr.intercept_
	print(f'''
Variable {nomVariable}, R2 = {100 * score:.1f} %
	Y = {p} * X + {q:.3f}''')

#~ ======================================================================
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(2 * 8.26, 1.3 * 8.26))

plt.rc('axes', labelsize=20) #fontsize of the x and y labels
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels

data = [ df[nomVariable].tolist() for nomVariable in df.columns ]
ax.boxplot(data, widths=0.6, showmeans=True, showfliers=False)
ax.set_title('Boxplots', fontsize=24)
plt.xticks(range(1, len(df.columns) + 1), df.columns)
plt.show()
plt.close(fig)

#~ ======================================================================
#~ Use StandardScaler to help you standardize the dataset’s features onto unit scale 
#~ (mean = 0 and variance = 1)
from sklearn.preprocessing import StandardScaler

_df = StandardScaler().fit_transform(df)
_df = DataFrame(data=_df, columns=df.columns)

#~ ======================================================================
fig, ax = plt.subplots(figsize=(2 * 8.26, 1.3 * 8.26))

plt.rc('axes', labelsize=20) #fontsize of the x and y labels
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels

data = [ _df[nomVariable].tolist() for nomVariable in _df.columns ]
ax.boxplot(data, widths=0.6, showmeans=True, showfliers=False)
ax.set_title('Boxplots (after StandardScaler)', fontsize=24)
plt.xticks(range(1, len(df.columns) + 1), df.columns)
plt.show()
plt.close(fig)

#~ ======================================================================
fig, ax = plt.subplots(figsize=(2 * 8.26, 1.3 * 8.26))

data = [ _df[nomVariable].tolist() for nomVariable in _df.columns ]
ax.violinplot(data, range(len(_df.columns)), points=30, widths=0.6,
	showextrema=True, showmeans=True, showmedians=True)

ax.set_title('Violin plots', fontsize=24)
plt.show()
plt.close(fig)

#~ ======================================================================
_df = concat([
	DataFrame(data=[{'Value': v, 'Variable': colname} for v in _df[colname]])
		for colname in df.columns
])

#~ ======================================================================
import seaborn as sns

plt.rc('axes', labelsize=20) #fontsize of the x and y labels
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels

fig, ax = plt.subplots(figsize=(2 * 8.26, 1.3 * 8.26))

sns.set_theme(style="whitegrid")
sns.violinplot(x='Variable', y='Value', data=_df, palette='coolwarm',
	inner='quartile')
ax.set_title('Violin plots', fontsize=24)
plt.show()
plt.close(fig)

#~ ======================================================================
#~ AFFICHAGE MODELES DE REGRESSION LINEAIRE
nrows, ncols = 1, len(nomsVariables)

fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*8.26, nrows*8.26))

for i, nomVariable in enumerate(nomsVariables):
	#~ JE REFAIS MON CALCUL DE REGRESSION 
	X = df[nomVariable].to_numpy().reshape(-1,1)
	y = df['Y']
	model = LinearRegression()
	model.fit(X, y)
	score, p, q = model.score(X, y), model.coef_[0], model.intercept_

	#~ J'ATTAQUE L'AFFICHAGE
	monEquationDeDroite = lambda x: p *x + q
	xTheorie = linspace(0.0, 1.0, 2) # 2 pts suffisent à définir une droite
	yTheorie = monEquationDeDroite(xTheorie)

	ax[i].set_title(f'scatter({nomVariable},Y): R2={100*score:.1f}%', fontsize=24)
	ax[i].plot(xTheorie, yTheorie, linestyle='-', color='red', linewidth=3,
		label=f'y $\simeq$ {p:.1f} x + {q:.1f}')
	ax[i].scatter(df[nomVariable], df.Y, label='Measure points')
	ax[i].set_xlabel(nomVariable, fontsize=20)
	ax[i].set_ylabel('Y', fontsize=20)
	ax[i].grid()
	ax[i].legend(loc='upper right', prop={'size': 20})

plt.show()
plt.close(fig)

#~ ======================================================================
