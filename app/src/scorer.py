import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import libs to solve classification task
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Make prediction
def make_pred(dt, path_to_file):

    print('Importing pretrained model...')
    # Import model
    model = CatBoostClassifier()
    model.load_model('./models/catboost_model.cbm')

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file, encoding='utf-8')['client_id'],
        'preds': (model.predict(dt)) * 1
    })
    print('Prediction complete!')

    return submission, model.predict_proba(dt)[:, 1]


def get_feature_importances(top_n=5):

    # Import model
    model = CatBoostClassifier()
    model.load_model('./models/catboost_model.cbm')

    # Getting the features and their importance
    feature_names = model.feature_names_
    features_importances = model.get_feature_importance()

    dict_feature_importances = dict(zip(feature_names, features_importances))
    sorted_dict_feature_importances = dict(sorted(dict_feature_importances.items(), key=lambda item: item[1], reverse=True)[:top_n])
    return sorted_dict_feature_importances

def plot_density_prediction_distribution(predictions, file_name='prediction_distribution.jpg'):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(predictions)
    plt.title('Плотность распределения предсказанных скоров')
    plt.xlabel('Предсказание')
    plt.ylabel('Плотность')
    plt.savefig(file_name)
    plt.close()
