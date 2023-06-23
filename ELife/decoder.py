from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneOut,
    RepeatedStratifiedKFold,
)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier


def get_pipeline(
    n_splits=10, penalty="l2", scoring="accuracy", scaler="standard", IF_BOOTS=0
):

    if n_splits == -1:
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

    print(cv)

    clf = LogisticRegressionCV(
        solver="liblinear",
        penalty=penalty,
        scoring=scoring,
        fit_intercept=True,
        cv=cv,
        n_jobs=None,
        verbose=0,
        class_weight="balanced",
    )

    pipe = []
    if scaler == "standard":
        pipe.append(("scaler", StandardScaler()))
    pipe.append(("clf", clf))

    pipe = Pipeline(pipe)

    if IF_BOOTS:
        pipe = BaggingClassifier(pipe, n_estimators=1000, n_jobs=-1)

    return pipe
