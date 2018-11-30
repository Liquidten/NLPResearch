import cuis_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,cross_validate




if __name__ == "__main__":
    logis = LogisticRegression()
    search_space = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "penalty": ["l1", "l2"]}
    grid_search = GridSearchCV(estimator=logis,
                               param_grid=search_space,
                               scoring="roc_auc",
                               cv=10)
    grid_search.fit(X=cuis_preprocess.X_train_matrix,
                    y=cuis_preprocess.y_train)
    print(grid_search.best_params_)
    logis.set_params(**grid_search.best_params_)
    score = cross_validate(estimator=logis,
                           X=cuis_preprocess.X_train_matrix,
                           y=cuis_preprocess.y_train.values,
                           scoring=["roc_auc", "precision","recall", "f1_micro", "f1_macro", "f1_weighted", "accuracy"],
                           cv=10,
                           return_train_score=False)
    ave_score = cuis_preprocess.dictMean(score)
    print(ave_score)
