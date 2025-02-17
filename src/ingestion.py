import os
from requests import get
from hashlib import md5
from time import time
from dotenv import load_dotenv
load_dotenv()

import pandas as pd


class MarvelIngestion(object):
    """
    [EN-US]
    The `MarvelIngestion` class is composed of the `get_params` function, which defines the variables, the hash and
    the parameters for the API request, the `__call__` method that makes the connection and the API call
    to extract the requested data.
    
    [PT-BR]
    A classe `MarvelIngestion` é composta pela função `get_params`, que define as variáveis, o hash e
    os parâmetros para a requisição à API, o método `__call__` que realiza a conexão e a chamada à API
    para extrair os dados solicitados.
    """
    def __init__(
        self,
        public_key,
        private_key,
        url='http://gateway.marvel.com/v1/public/',
        limit=100,
    ):
        """
        [EN-US]
        Initializes the class, defining the corresponding arguments.

        [PT-BR]
        Inicializa a classe, definindo os argumentos correspondentes.

        Args:
            public_key (str): the public key for connecting and using the APIs (a public key para conexão e uso das APIs).
            private_key (str): the private key for connecting and using the APIs (a private key para conexão e uso das APIs).
            url (str, optional): the Marvel Comics API’s base endpoint (O endpoint base da API da Marvel Comics).
            limit (int, optional): the requested result limit. Defaults to 100 (o limite de resultado solicitad. Padrão para 100).
        """
        super(MarvelIngestion, self).__init__()

        self.public_key = public_key
        self.private_key = private_key
        
        self.url = url
        self.limit = limit
        self.headers = {
            'Accept-Encoding': '*',
            'Accpet': '*/*',
            'Connection': 'keep-alive'
        }

    def get_params(self, offset, format_=None):
        """
        [EN-US]
        Defines the timestamp variable, the hash and the parameters for the API request.

        [PT-BR]
        Define a variável timestamp, o hash e os parâmetros para requisição à API.

        Agr:
            offset (int): The requested offset (skipped results) of the call
                          (O deslocamento solicitado (resultados ignorados) da chamada).
            format_ (str, optional): The publication format of the comic e.g. comic, hardcover, trade paperback. Default to None
                                     (O formato de publicação dos quadrinhos, por ex. comic, hardcover, trade paperback. Padrão para None).

        Return:
            params (dict): dictionary with the parameters for requesting the API containing (dicionário com os parâmetros para solicitação da API contendo):
                               ts: a timestamp (um timestamp).
                               apikey: your public key (sua public key).
                               hash: a md5 digest of the ts parameter, your private key and your public key
                                     (um digest md5 do parâmetro ts, da private key e da public key).
                               limit: the requested result limit (o limite de resultado solicitado).
                               offset: The requested offset (skipped results) of the call
                                       (O deslocamento solicitado (resultados ignorados) da chamada).
                               format: The publication format of the comic e.g. comic, hardcover, trade paperback. Default to None
                                       (O formato de publicação dos quadrinhos, por ex. comic, hardcover, trade paperback. Padrão para None).
        """
        ts = str(time())
        hash_ = md5(
            (
                ts + self.private_key + self.public_key
            ).encode('utf-8')
        ).hexdigest()
        params = {
                'ts': ts,
                'apikey': self.public_key,
                'hash': hash_,
                'limit': self.limit,
                'offset': offset,
                'format': format_
            }
        
        return params
    
    def __call__(self, endpoint, offset=0, format_=None, retries=5):
        """
        [EN-US]
        Connects to the API and makes the API call to extract the requested data.

        [PT-BR]
        Realiza a conexão com a API, realiza a chamada à API para a extração dos dados solicitados.
        
        Args:
            endpoint (str): endpoints to access the data, e.g. comics, characters (endpoints para acessar os dados, por ex. comics, characters).
            offset (int, optional): The requested offset (skipped results) of the call. Default to 0
                          (O deslocamento solicitado (resultados ignorados) da chamada. Padrão para 0).
            format_ (str, optional): The publication format of the comic e.g. comic, hardcover, trade paperback. Default to None
                                     (O formato de publicação dos quadrinhos, por ex. comic, hardcover, trade paperback. Padrão para None).
            retries (int, optional): attempts to connect, call and extract data. Default to 5
                                     (tentativas para a conexão, chamada e extração dos dados. Padrão para 5).

        Returns:
            df (pandas.DataFrame): data extracted into JSON transformed into a pandas DataFrame
                                   (dados extraídos em JSON transformados em um DataFrame do pandas).
        """        
        key = 'name' if endpoint == 'characters' else 'title'
        params = self.get_params(offset=offset, format_=format_)
        examples = []
        
        response = get(self.url + endpoint, params=params, headers=self.headers).json()
        total = response['data']['total']
                
        for page in range(offset, offset + total, self.limit):
            if retries > 0:
                try:
                    results = response['data']['results']
                    
                    for i in range(len(results)):
                        description = results[i]['description']
                        example = [
                                results[i]['id'],
                                results[i][key],
                                description
                            ]
                        if example not in examples and description and len(description) > 4:
                            examples.append(example)
                            
                    params = self.get_params(offset=page + self.limit, format_=format_)
                    response = get(self.url + endpoint, params=params, headers=self.headers).json()
                
                except:
                    retries -= 1
                    print(f'Error extracting data. Remaining attempts: {retries}')

        features = ['id', key, 'description']
        df = pd.DataFrame(examples, columns=features)

        return df


if __name__ == '__main__':
    PUBLIC_KEY = str(os.environ['MARVEL_PUBLIC_KEY'])
    PRIVATE_KEY = str(os.environ['MARVEL_PRIVATE_KEY'])

    PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir
        )
    )
    
    ingestion = MarvelIngestion(PUBLIC_KEY, PRIVATE_KEY)
    df_comics = ingestion(endpoint='comics', format_='comic', offset=0).drop_duplicates('description')
    
    
    labels = [
        'action',
        'non-action'
    ]
    
    corpus_comics = df_comics['description'].tolist()
    
    pipe_bart = pipeline(
        'zero-shot-classification',
        model='facebook/bart-large-mnli'
    )
    output_bart_comics = pipe_bart(corpus_comics, labels)
    
    labels_comic = []
    for i in range(len(output_bart_comics)):
        idx = np.argmax(output_bart_comics[i]['scores'])
        label = output_bart_comics[i]['labels'][idx]
        labels_comic.append(label)
    
    df_comics['y'] = labels_comic
    
    df_comics.to_csv(os.path.join('raw', 'comics_corpus.csv'), index=False)
