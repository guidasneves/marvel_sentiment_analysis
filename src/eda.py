import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud

import os


def get_dir():
    """
    
    """
    dir_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir
        )
    )

    return dir_path


def save_plot(name, format_='png', dpi=300):
    """
    
    """
    path = os.path.join(get_dir(), 'plots', name + '.' + format_)
    plt.savefig(path, format=format_, dpi=dpi)


def plot_bar(dataset, colors=None, name=None):
    """
    
    """
    labels_count = dataset['y'].value_counts()
    
    if not colors:
        colors = ['b', 'r']

    bar_plot = plt.bar(
        labels_count.index, 
        labels_count,
        color=colors,
        label=labels_count.index,
    )
    plt.bar_label(bar_plot, fmt='{:,.0f}')
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
    """
    
    """
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
    """
    
    """
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


if __name__ == '__main__':
    PATH = os.path.join(get_dir(), 'data')
    COLORS = ['cornflowerblue', 'chocolate']
    
    comics_data = pd.read_csv(os.path.join(PATH, 'raw', 'comics_corpus.csv'))
    print(f'Comics data shape: {comics_data.shape}')
    print('Comics dataset:\n', comics_data.isnull().sum())
    
    plot_bar(comics_data, colors=COLORS, name='comics_labels')
    
    index = 0
    label = 'action'
    
    comic_text = comics_data[comics_data['y'] == label]['description'].tolist()[index]
    print(f'Comic {index + 1} Label: {label}\n\nText {index + 1} example: {comic_text}')
    
    print(comics_data.iloc[:, 1:].describe().T)
    
    print(comics_data.info())
    
    comics_data['sentence_size'] = comics_data['description'].map(lambda x: len(x))
    comics_data['word_count'] = comics_data['description'].map(lambda x: len(x.split()))
    comics_data['capslock_word_count'] = comics_data['description'].map(lambda x: len(re.findall(r'\b[A-Z]+\b', x)))
    
    plot_hist_vs(comics_data, 'sentence_size', xlabel='Sentence size', bins=100, color=COLORS, name='sentence_size')
    plot_hist_vs(comics_data, 'word_count', xlabel='Word count', bins=50, color=COLORS, name='word_count')
    plot_hist_vs(comics_data, 'capslock_word_count', xlabel='Capslock word count', bins=50, color=COLORS, name='capslock_word_count')

    plot_wordcloud(comics_data, name='marvel_wordcloud')
