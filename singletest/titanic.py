import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib as mpl


from sklearn.model_selection import GridSearchCV

######################
# Helper
######################
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
 
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


######################
# Data Read
######################
print("#  Preparing Data")
data = pd.read_csv("data_titanic.csv")
data["Age"].fillna(data["Age"].mean(), inplace=True)
data["Embarked"].fillna(0, inplace=True)
data["Age"].astype(int)
data=data.sample(frac=1).reset_index(drop=True)
data['Age_cat'] = data['Age'].apply(lambda x: 1 if x > 18 else 0)
age = data[["Age","Survived"]]

######################
# Data Exploration
######################
print(data.head)
men = data.loc[(data.Sex == 0) & (data.Survived == 1)]["Survived"].count()
women = data.loc[(data.Sex == 1) & (data.Survived == 1)]["Survived"].count()
#print("Homme / Femme (percent) %i%%,%i%%" % (int(men*100), int(women * 100)))


print("##  Improving Data")
#Subdivision
mean_age = age.groupby(['Age'])['Survived'].mean().reset_index()
count_age = age.groupby(['Age'])['Survived'].sum().reset_index()
price = data[["Fare","Pclass"]].copy()
print("Rich women survival rate %i %%" % int(data.loc[(data.Sex==1) & (data.Pclass<2)]["Survived"].mean()*100))
print("Not rich women survival rate %i %%" % int(data.loc[(data.Sex==1) & (data.Pclass>1)]["Survived"].mean()*100))
rich = data.loc[(data.Pclass<2)]['Survived'].mean()
poor = data.loc[(data.Pclass>1)]['Survived'].mean()

print("###  Graph")
#Graph
fig, axs = plt.subplots(2, 3) 

axs[0,0].bar(mean_age["Age"],mean_age['Survived'])
axs[0,0].set_title("% de survie par age")
axs[0,1].bar(count_age["Age"],count_age['Survived'])
axs[0,1].set_title("Nombre de survivants par age")

axs[0,2].bar(price["Fare"], price["Pclass"])
axs[0,2].set_title("Price / Desk")



axs[1,1].bar(["1fst class", "others"],[rich, poor])
axs[1,1].set_title("% of survival per Class")


axs[1,0].pie([men,women],labels=["Men","Women"])
axs[1,0].set_title("% of survival per sex")
axs[1,2].pie([data.loc[data.Survived == 0]["Survived"].count(),data["Survived"].sum()],labels=["Dead", "Survived"])
axs[1,2].set_title("% of survival")

plt.show()

######################
# Model training
######################
train, test = train_test_split(data, test_size=0.1) 
y = train["Survived"]
features = [ "Age", "Sex", "sibsp", "Embarked"] #


# Show correlation map
fig, ax = plt.subplots()
correlation_matrix = train[features+["Survived"]].corr()
im, cbar = heatmap(correlation_matrix, features+["Survived"], features+["Survived"], ax=ax,
                   cmap="YlGn", cbarlabel="Heat map")
texts = annotate_heatmap(im, valfmt="{x:.1f} t")
plt.tight_layout()
plt.show()

print("####  Testing best parameters")
#Training
#X = pd.get_dummies(train[features])
#X_test = pd.get_dummies(test[features])

X = train[features]
X_test = test[features]

grid_clf = RandomForestClassifier()
param_grid = [
    {"n_estimators" : [10,100,200,500], "max_depth": [None, 5, 10, 30]}
]

grid_search = GridSearchCV(grid_clf, param_grid, cv = 5, scoring="accuracy", return_train_score=True)
grid_search.fit(X, y)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

print("#####  Using best models")
model = grid_search.best_estimator_
#model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], random_state=1)
model.fit(X, y)

print("\n\n###############################")
print("######    Final scoring  ######")
print("###############################")

#Inference
predictions = model.predict(X_test)

# Error Rate
error = (abs(test["Survived"] - predictions)).sum() / len(predictions)
print("Error percentage %i %%" % int(error*100))

importances = model.feature_importances_
forest_importances = pd.Series(importances, index=features)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("% of importance")
fig.tight_layout()


conf_mat = confusion_matrix(test["Survived"], predictions)
# Show correlation map
fig, ax = plt.subplots()
im, cbar = heatmap(conf_mat, [0,1], [0,1], ax=ax,
                   cmap="YlGn", cbarlabel="Heat map")
texts = annotate_heatmap(im, valfmt="{x:.1f} ")
plt.tight_layout()
plt.show()

#y_proba = model.predict_proba(X_test)