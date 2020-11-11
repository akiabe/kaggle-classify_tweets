import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    count_vec = CountVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None
    )
    count_vec.fit(df_train.text.values)

    xtrain = count_vec.transform(df_train.text.values)
    xvalid = count_vec.transform(df_valid.text.values)

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    lr = linear_model.LogisticRegression()
    lr.fit(xtrain, ytrain)
    pred = lr.predict_proba(xvalid)[:, 1]

    auc = metrics.roc_auc_score(yvalid, pred)
    print(f"fold={fold}, auc={auc}")

    df_valid.loc[:, "lr_cnt_pred"] = pred

    return df_valid[["id", "target", "kfold", "lr_cnt_pred"]]

if __name__ == "__main__":
    dfs = []
    for j in range(5):
        temp_df = run(j)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("../model_preds/lr_cnt.csv", index=False)