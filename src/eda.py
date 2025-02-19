# Packages used in the system
import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
plt.rcParams['figure.figsize'] = (8, 8) # set default size of plots

import os


def get_dir():
    """
    [EN-US]
    Returns the path of the project's root directory.

    [PT-BR]
    Retorna o path do diretório raiz do projeto.

    Returns:
        dir_path (str): path of the project root directory.
    """
    # Getting Obtaining the absolute normalized version of the project root path
    dir_path = os.path.abspath(
        os.path.join( # Concatenating the paths
            os.path.dirname(__file__), # file path
            os.pardir # Gettin the constant string to refer to the parent directory
        )
    )

    return dir_path


def save_plot(name, format_='png', dpi=300):
    """
    [EN-US]
    Saves the plot with the specified name, format and dpi.
    
    [PT-BR]
    Salva o plot com o nome, formato e dpi específicado.
    
    Argument:
        name (str): name that the plot will be saved.
        format_ (str, optional): format in which the plot will be saved. Default to png.
        dpi (int, optional): dpi that the plot will be saved. Default to 300.
    """
    # Setting the directory and name in which the plot will be saved (Definindo o diretório e o nome que o plot será salvo)
    path = os.path.join(get_dir(), 'plots', name + '.' + format_)
    # Saving the plot (Salvando o plot)
    plt.savefig(path, format=format_, dpi=dpi)


def get_word_frequency(dataset, size=10, drop_stopwords=False):
    """
    [EN-US]
    Returns the path of the project's root directory.

    [PT-BR]
    Retorna o path do diretório raiz do projeto.

    Argument:
        data (pandas.DataFrame): name that the plot will be saved.
        size (int, optional): 
        drop_stopwords (bool, optional): 

    Returns:
        dir_path (str): path of the project root directory.
    """
    corpus = ' '.join(dataset['description'].tolist())
    if drop_stopwords:
        corpus = [word for word in corpus.split() if word.lower() not in stopwords.words('english')]
    word_frequency = Counter(corpus)
    
    action_corpus = ' '.join(dataset[dataset['y'] == 'action']['description'].tolist())
    wf_action = Counter(action_corpus.split())
    wf_n_action = word_frequency - wf_action
        
    keys, wf_action_l, wf_n_action_l = [], [], []
    for i, _ in word_frequency.most_common(size):
        keys.append(i)
        wf_action_l.append(wf_action[i])
        wf_n_action_l.append(wf_n_action[i])

    return keys, wf_action_l, wf_n_action_l


def plot_bar(dataset, colors=None, name=None):
    
    labels_count = dataset['y'].value_counts()
    if not colors:
        colors = ['b', 'r']

    bar_plot = plt.bar(
        labels_count.index, 
        labels_count,
        color=colors,
        label=labels_count.index,
    )
    plt.bar_label(bar_plot, fmt='{:,.0f}', fontsize=16)
    plt.title('Comics Dataset Labels', fontsize=16)
    plt.xlabel('Labels', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    if name:
        save_plot(name)
    plt.show()


def plot_hist_vs(dataset,
                 feature, 
                 alpha=.8, 
                 xlabel=None, 
                 bins=100, 
                 colors=None, 
                 name=None):
    
    if not colors:
        colors = ['b', 'r']
    
    plt.hist(dataset[dataset['y'] == 'action'][feature], label='Action', color=colors[0], bins=bins)
    plt.hist(dataset[dataset['y'] != 'action'][feature], label='Non-Action', alpha=alpha, color=colors[1], bins=bins)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    if name:
        save_plot(name)
    plt.show()


def plot_wordcloud(dataset, cmap='gist_heat', name=None):
    
    corpus = ' '.join(dataset['description'].tolist())
    mask = np.array(
        Image.open(
            os.path.join(get_dir(), 'figures', 'marvel_logo.png')
            )
    )

    wc = WordCloud(
        background_color='white',
        mask=mask,
        stopwords=stopwords.words('english') + ['Marvel'],
        width=1000,
        height=600,
        colormap=cmap,
        contour_width=2,
        contour_color='gray'
    ).generate(corpus)
    plt.axis('off')
    plt.imshow(wc)
    if name:
        save_plot(name)


def plot_word_frequency(dataset, size=10, drop_stopwords=False, colors=None, name=None):
    
    width = .25
    x = np.array(range(size))
    x_shifted = x + width
    keys, wf_action, wf_n_action = get_word_frequency(dataset, size, drop_stopwords)
    if not colors:
        colors = ['b', 'r']

    plt.bar(x, wf_action, width=width, color=colors[0], label='Action', edgecolor='white')
    plt.bar(x_shifted, wf_n_action, width=width, color=colors[1], label='Non-action', edgecolor='white')
    plt.xticks(range(size), keys)
    plt.title('Most common words frequency', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlabel('Words', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    if name:
        save_plot(name)
    plt.show()


if __name__ == '__main__':
    # Setting the global variable with the path of the directory where the data will be loaded
    PATH = os.path.join(get_dir(), 'data')
    
    # Reading the dataset from the `../data/raw/` directory and checking its size
    comics_data = pd.read_csv(os.path.join(PATH, 'raw', 'comics_corpus.csv'))
    print(f'Comics data shape: {comics_data.shape}')
    # Counting null values
    print(f'Null values in comics dataset:\n{comics_data.isnull().sum()}')

    # Setting the global variable with the colors for the plots that will be created later
    COLORS = ['cornflowerblue', 'chocolate']
    
    # Examining the `index` example for the `label` label examples
    index = 0
    label = 'action'
    comic_text = comics_data[comics_data['y'] == label]['description'].tolist()[index]
    print(f'Comic {index + 1} Label: {label}\n\nText {index + 1} example: {comic_text}')
    
    # Performing descriptive analysis. We perform descriptive analysis to help identify problems
    print(comics_data.iloc[:, 1:].describe().T)
    # Analyzing the type of each feature and whether there are null values
    print(comics_data.info())
    
    # Plotting the distribution and count between the 2 labels, the `action` and the `non-action`
    plot_bar(comics_data, colors=COLORS, name='comics_labels')
    # Plotting a wordcloud with the most frequent words in the corpus. The stopwords were excluded, so that we could only see the words that really have importance and meaning
    plot_wordcloud(comics_data, name='marvel_wordcloud')
    # Plotting the frequency of the words that appear most in the corpus, except stopwords, divided between the labels `action` and `non-action`
    plot_word_frequency(comics_data, size=10, colors=COLORS, drop_stopwords=True, name='word_frequency')

    # Creating some features to help with the plots that will be created next
    comics_data['sentence_size'] = comics_data['description'].map(lambda x: len(x) - x.count(' '))
    comics_data['word_count'] = comics_data['description'].map(lambda x: len(x.split()))
    comics_data['capslock_word_count'] = comics_data['description'].map(lambda x: len(re.findall(r'\b[A-Z]+\b', x)))
    comics_data['unique_word_count'] = comics_data['description'].map(lambda x: len(set(x.split())))
    
    # Plotting the comparison of the distribution between the `action` label and the `non-action` label in the features that were created
    plot_hist_vs(comics_data, 'sentence_size', xlabel='Sentence size', bins=100, color=COLORS, name='sentence_size')
    plot_hist_vs(comics_data, 'word_count', xlabel='Word count', bins=50, color=COLORS, name='word_count')
    plot_hist_vs(comics_data, 'capslock_word_count', xlabel='Capslock word count', bins=50, color=COLORS, name='capslock_word_count')
    plot_hist_vs(comics_data, 'unie_word_count', xlabel='Unique word count', bins=40, color=COLORS, name='unique_word_count')
