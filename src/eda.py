# Packages used in the system
# Pacotes utilizados no sistema
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
        dir_path (str): path of the project root directory
                        (caminho do diretório raiz do projeto).
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
    Saves the plot in the `../plots/` directory with the specified name, format and dpi.
    
    [PT-BR]
    Salva o plot no diretório `../plots/` com o nome, formato e dpi específicado.
    
    Arguments:
        name (str): name that the plot will be saved
                    (nome que o gráfico será salvo).
        format_ (str, optional): format in which the plot will be saved. Default to png
                                 (formato no qual o gráfico será salvo. Padrão para png).
        dpi (int, optional): dpi that the plot will be saved. Default to 300
                             (dpi que o gráfico será salvo. Padrão para 300).
    """
    # Setting the directory and name in which the plot will be saved
    # Definindo o diretório e o nome que o plot será salvo
    path = os.path.join(get_dir(), 'plots', name + '.' + format_)
    # Saving the plot
    # Salvando o plot
    plt.savefig(path, format=format_, dpi=dpi)


def get_word_frequency(dataset, size=10, drop_stopwords=False):
    """
    [EN-US]
    Receives a pandas DataFrame and creates a `size` frequency dictionary of the words that occur most in the corpus between the positive and negative classes.

    [PT-BR]
    Recebe um DataFrame do pandas e cria um dicionário de frequência `size` das palavras que mais ocorrem no corpus entre as classes positivas e negativas.

    Arguments:
        data (pandas.DataFrame): pandas DataFrame to create the word frequency dictionary
                                 (DataFrame do pandas para criar o dicionário de word frequency). 
        size (int, optional): number of words in the dictionary with the highest frequency. Defaults to 10
                              (quantidade de palavras no dicionário com a maior frequência. Padrão para 10).
        drop_stopwords (bool, optional): drop stopwords if True, else do not drop. Defaults to False
                                         (exclui as stopwords se True, else não exluci. Padrão para False)

    Returns:
        dir_path (str): path of the project root directory
                        (caminho do diretório raiz do projeto).
    """
    # Transforming the feature into a single corpus
    # Transformando a feature em um corpus único
    corpus = ' '.join(dataset['description'].tolist())
    # Dropping stopwords if True
    # Excluindo as stopwords se True
    if drop_stopwords:
        corpus = [word for word in corpus.split() if word.lower() not in stopwords.words('english')]
    # Creating the dictionary with word frequencies
    # Criando o dicionário com as word frequencies
    word_frequency = Counter(corpus)

    # Transforming only the phrases from the `action` label into a single corpus
    # Transformando apenas as frases do label `action` em um corpus único
    action_corpus = ' '.join(dataset[dataset['y'] == 'action']['description'].tolist())
    # Creating the dictionary with word frequencies only from the `action` label corpus
    # Criando o dicionário com as word frequencies apenas do corpus do label `action`
    wf_action = Counter(action_corpus.split())
    # Creating the dictionary with word frequencies only from the `non-action` label corpus
    # Criando o dicionário com as word frequencies apenas do corpus do label `non-action`
    wf_n_action = word_frequency - wf_action
        
    keys, wf_action_l, wf_n_action_l = [], [], []
    # Looping through the most frequent size words to return
    # Percorrendo as size palavras mais frequentes para retornar
    for i, _ in word_frequency.most_common(size):
        # Adding the word
        # Adicionando o palavra
        keys.append(i)
        # Adding the frequency of this word in the `action` label corpus
        # Adicionando a frequência dessa palavra no corpus do label `action`
        wf_action_l.append(wf_action[i])
        # Adding the frequency of this word in the `non-action` label corpus
        # Adicionando a frequência dessa palavra no corpus do label `non-action`
        wf_n_action_l.append(wf_n_action[i])

    return keys, wf_action_l, wf_n_action_l


def plot_bar(dataset, colors=None, name=None):
    """
    [EN-US]
    Returns a histogram comparing the distribution of numeric features between the positive and negative class.
    
    [PT-BR]
    Retorna um histograma comparando a distribuição das features numéricas entre as classes positiva e negativa.
    
    Arguments:
        data (pandas.DataFrame): pandas DataFrame to create the label distribution plot
                                 (DataFrame do pandas para criar o plot da distribuição dos labels).
        colors (list, optional): list with 2 colors arguments for the plot. Defaults to None
                                 (lista com 2 cores argumentos para o plot. Padrão para None).
        name (str, optional): name that the plot will be saved. Defaults to None
                              (nome que o gráfico será salvo. Padrão para None).
    """
    labels_count = dataset['y'].value_counts()
    # If a list with 2 colors is not passed, it will use the ones defined below
    # Se uma lista com 2 cores não for passada, usará as definidas abaixo
    if not colors:
        colors = ['b', 'r']

    # Setting the plot
    # Definindo o plot
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
    # If a name is passed, the plot is saved
    # Se algum nome for passado o plot é salvo
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
    [EN-US]
    Returns a histogram comparing the distribution of numeric features between the positive and negative class.
    
    [PT-BR]
    Retorna um histograma comparando a distribuição das features numéricas entre as classes positiva e negativa.
    
    Arguments:
        data (pandas.DataFrame): pandas DataFrame to create the label distribution plot
                                 (DataFrame do pandas para criar o gráfico de distribuição de rótulos).
        alpha (float, optional): alpha value used for blending. alpha must be within the 0-1 range, inclusive. Defaults to 0.8
                                 (valor alpha usado para mesclagem. alpha deve estar dentro do intervalo 0-1, inclusive. Padrão para 0,8).
        xlabel (str, optional): set the label for the x-axis. Defaults to None
                                (define o rótulo para o eixo x. Padrão para None).
        bins (int, optional): it defines the number of equal-width bins in the range. Defaults to 100
                              (ele define o número de caixas de largura igual no intervalo. Padrão para 100).
        colors (list, optional): list with 2 colors arguments for the plot. Defaults to None
                                 (lista com 2 cores argumentos para o plot. Padrão para None).
        name (str, optional): name that the plot will be saved. Defaults to None
                              (nome que o gráfico será salvo. Padrão para None).
    """
    # If a list with 2 colors is not passed, it will use the ones defined below
    # Se uma lista com 2 cores não for passada, usará as definidas abaixo
    if not colors:
        colors = ['b', 'r']

    # Setting the plot
    # Definindo o plot
    plt.hist(dataset[dataset['y'] == 'action'][feature], label='Action', color=colors[0], bins=bins)
    plt.hist(dataset[dataset['y'] != 'action'][feature], label='Non-Action', alpha=alpha, color=colors[1], bins=bins)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    # If a name is passed, the plot is saved
    # Se algum nome for passado o plot é salvo
    if name:
        save_plot(name)
    plt.show()


def plot_wordcloud(dataset, cmap='gist_heat', name=None):
    """
    [EN-US]
    Returns a histogram comparing the distribution of numeric features between the positive and negative class.
    
    [PT-BR]
    Retorna um histograma comparando a distribuição das features numéricas entre as classes positiva e negativa.
    
    Arguments:
        data (pandas.DataFrame): pandas DataFrame to create word cloud plot
                                 (DataFrame do pandas para criar o plot da nuvem de palavras).
        cmap (str, optional): the plot colormap. Defaults to gist_heat
                              (o colormap do plot. Padrão para gist_heat).
        name (str, optional): name that the plot will be saved. Defaults to None
                              (nome que o gráfico será salvo. Padrão para None).
    """
    # Transforming the feature into a single corpus
    # Transformando a feature em um corpus único
    corpus = ' '.join(dataset['description'].tolist())
    # Setting the image that will be used as a mask
    # Definindo a imagem que será utilizada como máscara
    mask = np.array(
        Image.open(
            os.path.join(get_dir(), 'figures', 'marvel_logo.png')
            )
    )

    # Setting the wordcloud plot
    # Definindo o wordcloud plot
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
    # If a name is passed, the plot is saved
    # Se algum nome for passado o plot é salvo
    if name:
        save_plot(name)


def plot_word_frequency(dataset, size=10, drop_stopwords=False, colors=None, name=None):
    """
    [EN-US]
    Returns a histogram comparing the distribution of numeric features between the positive and negative class.
    
    [PT-BR]
    Retorna um gráfico de barras comparando a distribuição das palavras mais frequentes entre as classes positiva e negativa.
    
    Arguments:
        data (pandas.DataFrame): data (pandas.DataFrame): pandas DataFrame to create the word frequency dictionary and the word frequency bar plot
                                 (DataFrame do pandas para criar o dicionário de word frequency e o plot de barras de word frequency).
        size (int, optional): number of words in the dictionary with the highest frequency. Defaults to 10
                              (quantidade de palavras no dicionário com a maior frequência. Padrão para 10).
        drop_stopwords (bool, optional): drop stopwords if True, else do not drop. Defaults to False
                                         (exclui as stopwords se True, else não exluci. Padrão para False)
        colors (list, optional): list with 2 colors arguments for the plot. Defaults to None
                                 (lista com 2 cores argumentos para o plot. Padrão para None).
        name (str, optional): name that the plot will be saved. Defaults to None
                              (nome que o gráfico será salvo. Padrão para None).
    """
    width = .25
    # Defining the size axis size
    # Definindo o eixo do tamanho size
    x = np.array(range(size))
    # Shifting the second axis by width
    # Deslocando o segundo eixo em width
    x_shifted = x + width
    # Get the word frequencies for both labels and each corresponding word
    # Obtém as word frequencies para ambos os labels e cada palavra correspondente
    keys, wf_action, wf_n_action = get_word_frequency(dataset, size, drop_stopwords)
    # If a list with 2 colors is not passed, it will use the ones defined below
    # Se uma lista com 2 cores não for passada, usará as definidas abaixo
    if not colors:
        colors = ['b', 'r']

    # Setting the plot
    # Definindo o plot
    plt.bar(x, wf_action, width=width, color=colors[0], label='Action', edgecolor='white')
    plt.bar(x_shifted, wf_n_action, width=width, color=colors[1], label='Non-action', edgecolor='white')
    plt.xticks(range(size), keys)
    plt.title('Most common words frequency', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlabel('Words', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    # If a name is passed, the plot is saved
    # Se algum nome for passado o plot é salvo
    if name:
        save_plot(name)
    plt.show()


if __name__ == '__main__':
    # Setting the global variable with the path of the directory where the data will be loaded
    # Configurando a variável global com o path do diretório onde os dados serão carregados
    PATH = os.path.join(get_dir(), 'data')
    
    # Reading the dataset from the `../data/raw/` directory and checking its size
    # Lendo o dataset do diretório `../data/raw/` e verificando seu tamanho
    comics_data = pd.read_csv(os.path.join(PATH, 'raw', 'comics_corpus.csv'))
    print(f'Comics data shape: {comics_data.shape}')
    # Counting null values
    # Contando valores nulos
    print(f'Null values in comics dataset:\n{comics_data.isnull().sum()}')

    # Setting the global variable with the colors for the plots that will be created later
    # Configurando a variável global com as cores para os gráficos que serão criados posteriormente
    COLORS = ['cornflowerblue', 'chocolate']
    
    # Examining the `index` example for the `label` label examples
    # Examinando o exemplo `index` para os exemplos de rótulos `label`
    index = 0
    label = 'action'
    comic_text = comics_data[comics_data['y'] == label]['description'].tolist()[index]
    print(f'Comic {index + 1} Label: {label}\n\nText {index + 1} example: {comic_text}')
    
    # Performing descriptive analysis. We perform descriptive analysis to help identify problems
    # Realizando a análise descritiva. Realizamos análises descritivas para ajudar a identificar problemas
    print(comics_data.iloc[:, 1:].describe().T)
    # Analyzing the type of each feature and whether there are null values
    # Analisando o tipo de cada feature e se existem valores nulos
    print(comics_data.info())
    
    # Plotting the distribution and count between the 2 labels, the `action` and the `non-action`
    # Plotando a distribuição e a contagem entre os 2 rótulos, a `action` e a `non-action`
    plot_bar(comics_data, colors=COLORS, name='comics_labels')
    # Plotting a wordcloud with the most frequent words in the corpus. The stopwords were excluded, so that we could only see the words that really have importance and meaning
    # Plotando uma nuvem de palavras com as palavras mais frequentes do corpus. As stopwords foram excluídas, para visualizarmos apenas as palavras que realmente têm importância e significado
    plot_wordcloud(comics_data, name='marvel_wordcloud')
    # Plotting the frequency of the words that appear most in the corpus, except stopwords, divided between the labels `action` and `non-action`
    # Plotando a frequência das palavras que mais aparecem no corpus, exceto stopwords, divididas entre os rótulos `action` e `non-action`
    plot_word_frequency(comics_data, size=10, colors=COLORS, drop_stopwords=True, name='word_frequency')

    # Applying feature engineering to help with the plots that will be created next
    # Aplicando feature engineering para ajudar nas plotagens que serão criadas a seguir
    comics_data['sentence_size'] = comics_data['description'].map(lambda x: len(x) - x.count(' '))
    comics_data['word_count'] = comics_data['description'].map(lambda x: len(x.split()))
    comics_data['capslock_word_count'] = comics_data['description'].map(lambda x: len(re.findall(r'\b[A-Z]+\b', x)))
    comics_data['unique_word_count'] = comics_data['description'].map(lambda x: len(set(x.split())))
    
    # Plotting the comparison of the distribution between the `action` label and the `non-action` label in the features that were created
    # Plotando a comparação da distribuição entre o rótulo `action` e o rótulo `non-action` nas features que foram criados
    plot_hist_vs(comics_data, 'sentence_size', xlabel='Sentence size', bins=100, color=COLORS, name='sentence_size')
    plot_hist_vs(comics_data, 'word_count', xlabel='Word count', bins=50, color=COLORS, name='word_count')
    plot_hist_vs(comics_data, 'capslock_word_count', xlabel='Capslock word count', bins=50, color=COLORS, name='capslock_word_count')
    plot_hist_vs(comics_data, 'unie_word_count', xlabel='Unique word count', bins=40, color=COLORS, name='unique_word_count')
