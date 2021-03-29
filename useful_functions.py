import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

def model_fitter(model, X_train, y_train, X_test, y_test, cv=5):
    '''
    Function to fit a model, print out relevant scores
    and return the scores in a tuple.
    '''
    model.fit(X_train, y_train)
    cv_score = cross_val_score(model, X_train, y_train, cv=cv).mean()
    training_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print('Mean cross-validated training score: ', cv_score)
    print('Training score: ', training_score)
    print('Test score: ', test_score)
    return cv_score, training_score, test_score

def gridsearch_model_fitter(model, X_train, y_train, X_test, y_test):
    '''
    Function to fit a model, print out relevant scores
    and return the scores in a tuple.
    '''
    model.fit(X_train, y_train)
    cv_score = model.best_score_
    training_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print('Best parameters: ', model.best_params_)
    print('Best estimator mean cross-validated training score: ', cv_score)
    print('Best estimator score on the training set: ', training_score)
    print('Best estimator score on the test set: ', test_score)
    return cv_score, training_score, test_score

def plot_coefficients(model, feature_names, n_to_plot=10):
    '''
    Function to collect model coefficients in a DataFrame
    and then plot the n most important ones. Function
    returns the DataFrame and the plot.
    '''
    # collect model coefficients in a DataFrame
    df_coef = pd.DataFrame({'variable': feature_names,
                            'coef': model.coef_, 'coef_abs': np.abs(model.coef_)})
    df_coef.sort_values('coef_abs', ascending=False, inplace=True)
    # plot coefficients from model
    fig, ax = plt.subplots(figsize=(14,8))
    sns.barplot(data=df_coef.head(n_to_plot), x='coef', y='variable')
    plt.title(f'The {n_to_plot} coefficients from the model with the most impact')
    plt.show()
    return df_coef, fig

def plot_model_scores(df, scores1_col, scores2_col, models_col,
                      scores1_label, scores2_label, y_min=0.5, y_max=0.95):
    '''
    Function to plot two sets of model scores from a DataFrame,
    ordered by scores1.
    '''
    plot_df = df.sort_values(by=scores1_col, ascending=False)
    labels = plot_df[models_col]
    scores1 = round(plot_df[scores1_col], 3)
    scores2 = round(plot_df[scores2_col], 3)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 10))
    rects1 = ax.bar(x - width/2, scores1, width, label=scores1_label)
    rects2 = ax.bar(x + width/2, scores2, width, label=scores2_label)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of model scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, fontsize=14)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    def autolabel(rects):
        '''
        Attach a text label above each bar in *rects*, displaying its height.
        '''
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12)

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()
