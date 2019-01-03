import pandas as pd
import numpy as np
from collections import Counter

def preprocess(train, test):
    # Age の欠損値を、中央値で埋める
    trn_ag = train["Age"]
    trn_ag_med = trn_ag[trn_ag.notnull()].median()
    tst_ag = test["Age"]
    tst_ag_med = tst_ag[tst_ag.notnull()].median()
    train["Age"] = trn_ag.fillna(trn_ag_med)
    test["Age"] = tst_ag.fillna(tst_ag_med)

    # Fare の欠損値を、中央値で埋める (test のみ)
    tst_fr = test["Fare"]
    tst_fr_med = tst_fr[tst_fr.notnull()].median()
    test["Fare"] = tst_fr.fillna(tst_fr_med)

    # Embarked の欠損値を、最頻値で埋める
    trn_mb = train["Embarked"]
    trn_mb_mc = Counter(trn_mb[trn_mb.notnull()]).most_common(1)[0][0]
    tst_mb = test["Embarked"]
    tst_mb_mc = Counter(tst_mb[tst_mb.notnull()]).most_common(1)[0][0]
    train["Embarked"] = trn_mb.fillna(trn_mb_mc)
    test["Embarked"] = tst_mb.fillna(tst_mb_mc)

    # Sex を数値で表現する (male->0, female->1)
    train["Sex"] = [0 if s == "male" else s for s in train["Sex"]]
    train["Sex"] = [1 if s == "female" else s for s in train["Sex"]]
    test["Sex"] = [0 if s == "male" else s for s in test["Sex"]]
    test["Sex"] = [1 if s == "female" else s for s in test["Sex"]]

    # Embarked を数値で表現する (S->0, C->1, Q->2)
    train["Embarked"] = [0 if e == "S" else e for e in train["Embarked"]]
    train["Embarked"] = [1 if e == "C" else e for e in train["Embarked"]]
    train["Embarked"] = [2 if e == "Q" else e for e in train["Embarked"]]
    test["Embarked"] = [0 if e == "S" else e for e in test["Embarked"]]
    test["Embarked"] = [1 if e == "C" else e for e in test["Embarked"]]
    test["Embarked"] = [2 if e == "Q" else e for e in test["Embarked"]]
