import pandas as pd
import numpy as np

def processing_data():
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'] # Definindo nomes das features
    df = pd.read_csv('data/auto-mpg.data', sep=r'\s+', names=columns) # Importando os dados de treinamento em um DataFrame
    
    df['origin'] = df['origin'].map({1: 'USA', 2:'Europe', 3:'Japan'}) # Alterando valores da coluna 'Origem'
    df = pd.get_dummies(df, columns=['origin'], dtype='int64') # Transformando valores da coluna 'Origem' em variáveis indicadoras

    df.drop(columns=['car_name', 'acceleration'], inplace=True) # Removendo colunas não utilizáveis

    df['sqr_horsepower'] = df['horsepower']**2 # Adicionando novas features
    df['cube_horsepower'] = df['horsepower']**3 # Adicionando novas features
    df['sqr_weight'] = df['weight']**2 # Adicionando novas features
    df['cube_weight'] = df['weight']**3 # Adicionando novas features
    df['prod_sqr_horsepower'] = df['horsepower'] * df['sqr_horsepower'] # Adicionando novas feature
    df['prod_sqr_weight'] = df['weight'] * df['sqr_weight'] # Adicionando novas feature
    return df

def conversion_data(df):
    df['km_l'] = df['mpg'] * 0.425144 # Convertendo os valores target de (Milhas p/ Galão) para (Km/L)

    Y = df['km_l'].values # Convertendo os targes do DF em um nparray
    X = df.drop(columns=['mpg', 'km_l']).values # Convertendo as Features em um nparray

    return Y, X

def zscore_normalization(x):
    '''
    Dimensiona/normaliza as features 

    Args:
        x (ndarray (m, n)): Features não normalizadas
    Retornos:
        x_norm (ndarray (m, n)): Features normalizadas
        mu (ndarray (n, ))     : Média das features para ser utilizada em normalizações futuras
        sigma (ndarray (n, ))  : Desvio padrão das features para ser utilizado em normalizações futuras
    '''

    mu = np.mean(x, axis=0) # Média de cada feature
    sigma = np.std(x, axis=0) # Desvio padrão de cada feature

    x_norm = (x - mu) / sigma # Normalização das features
    return x_norm, mu, sigma

def compute_cost(x, y, w, b, lambda_):
    '''
    Função para calcular o custo (Mean Squarred Error [MSE])
    Args:
        X (ndarray (m, n)): Dados de treinamento, m exemplos com n features
        Y (ndarray  (m, )): Targets
        w (ndarray  (n, )): Parâmetro do modelo
        b (scalar)        : Parâmetro do modelo

    Retorno:
        cost (scalar): Custo
    '''

    m,n = x.shape # 'm' recebe o número de conjuntos de treinmamento
    cost = 0.0

    for i in range(m): # Representação em código da função de custo
        f_wb = np.dot(x[i], w) + b
        cost = cost + (f_wb - y[i])**2
    cost = cost / (2 * m)

    reg_cost = 0 # Adicionando termo de regularização
    for j in range(n): # Representação em código do termo de regularização
        reg_cost += (w[j]**2)
    reg_cost = (lambda_ / (2 * m)) * reg_cost

    total_cost = cost + reg_cost
    return total_cost

def compute_gradient(x, y, w, b, lambda_):
    '''
    Função para calcular as derivadas dos parâmetros do modelo
    Args:
        X (ndarray (m, n)): Dados de treinamento, m exemplos com n features
        Y (ndarray  (m, )): Targets
        w (ndarray  (n, )): Parâmetro do modelo
        b (scalar)        : Parâmetro do modelo

    Retorno:
        dj_dw (ndarray (n,)): Derivada do parâmetro "w"
        dj_db (scalar)      : Derivada do parâmetro "b"
    '''
    m, n = x.shape # 'm' e 'n' recebem a quantidade de conjunstos de treinamento e a quantidade de features, respectivamente
    dj_dw = np.zeros(n) 
    dj_db = 0

    for i in range(m): # Representação em código do cálculo das derivadas dos parâmetros de modelo
        dif = (np.dot(x[i], w) + b) - y[i]
        for j in range(n): 
            dj_dw[j] = dj_dw[j] + dif * x[i, j] # Atualização simultânea 
        dj_db = dj_db + dif # Atualização simultânea 
    
    dj_dw = dj_dw  / m
    dj_db = dj_db / m

    for j in range(n): # Adição do termo de regularização
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j] 
    
    return dj_dw, dj_db

def save_cost(cost_funcion, i, x, y, w, b, lambda_):
    '''
    Função que retorna o custo e é chamada a cada iteração do algoritmo de descida de gradiente

    Args:
        cost_func            : Função que calcula o custo (Mean Squarred Error [MSE])
        i (int)              : Iteração atual
        x, y, w, b, lambda_  : Parâmetros que fazem a função de custo funcionar

    Retorno:
        cost (float) : Custo da iteração atual
    '''

    cost = cost_funcion(x, y, w, b, lambda_) # A variável custo recebe o valor do custo da iteração atual
    
    if i<100000: # A função retorna o custo caso o número de iterações não ultrapasse o limite definido
        return cost
    else:
        return None

def gradient_descent(x, y, w_in, b_in, gradient_func, cost_func, alpha, lambda_, num_iters):
    '''
    Função que treina o modelo para encontrar os melhores valores para os parâmetros 'w' e 'b'

    Args:
        X (ndarray    (m, n)): Dados de treinamento, m exemplos com n features
        Y (ndarray     (m, )): Targets
        w_in (ndarray  (n, )): Parâmetro do modelo, valor inicial
        b_in (scalar)        : Parâmetro do modelo, valor inicial
        cost_func            : Função para calcular o custo    
        gradient_func        : Função para calcular as derivadas
        alpha (float)        : Taxa de aprendizade
        nun_iters (int)      : Número de iterações

    Retorno:
        w (ndarray (n,)) : Valor atualizado de 'w'
        b (scalar)       : Valor atualizado de 'b'
    '''
    print("Training the Model. Please wait!")

    J_history = {} # Dicionário que guarda o histórico do valor do custo (MSE)

    w = w_in # Inicialização dos parâmetros W (Coeficientes)
    b = b_in # Inicialização do parâmetro b (Intercepto)

    for i in range(num_iters): # Representação em código do algoritmo de descida de gradiente
        dj_dw, dj_db = gradient_func(x, y, w, b, lambda_)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_history[i + 1] = save_cost(cost_func, i, x, y, w, b, lambda_) # O valor do custo é registrado
    
    print("Training completed!\n")
    return J_history, w, b

def prediction(x, w, b):
    '''
    Função que prevê um valor, a partir de features desconhecidas e após o modelo ser treinado

    Args:
        X (ndarray (n,)): Features
        w (ndarray (n,)): Parãmetro do modelo
        b (scalar)      : Parãmetro do modelo

    Retorno:
        predict (float): Valor previsto
    '''
    predict = np.dot(x, w) + b # Representação em código do cálculo da regressão linear
    return predict

def training_model():
    '''
    Função que treina o modelo com os dados fornecidos
    Args:
        df (DataFrame (m. n)): DataFrame com todos os dados de treinamento. 'm' Conjuntos e 'n' Features

    Retorno:
        w_final (ndarray (n,)): Parâmetro 'w' após treinamento
        b_final (scalar)      : Parâmetro 'b' após treinamento
    '''
    df = processing_data() # Puxando todos os dados já tratados e prontos para treinamento

    y_train, x_train = conversion_data(df) # Separando os dados em features e targets
    x_train, mu, sigma = zscore_normalization(x_train) # Normalizando as features e guardando os valores da média e do desvio padrão

    w_init = np.zeros(x_train.shape[1]) # Inicialização dos parâmetros W (Coeficientes)
    b_init = 0 # Inicialização dos parâmetros b (Intercepto)

    J_history = {} # Dicionário que vai receber o registro do custo (MSE)
    iterations = 2000 # Número de iterações que o algoritmo de descida de gradiente vai realizar
    alpha = 0.03 # Taxa de aprendizado
    lambda_ = 0.5 # Taxa de regularização

    J_history, w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, compute_gradient, compute_cost, alpha, lambda_, iterations)
    return J_history, w_final, b_final, mu, sigma


def main():
    J_history = {} # Dicionário que vai receber o registro do custo (MSE)
    J_history, w, b, mu, sigma = training_model() # Registro do custo, coeficientes, intercepto, média das features e desvio padrão das features recebem seus valores finais
    
    # Aplicação
    print('=============== CarFuel Regression Model (Without Sklearn) ===============')
    while True:
        x_prev = {
                'cylinders': 0,
                'displacement': 0,
                'horsepower': 0,
                'weight': 0,
                'model_year': 0,
                'origin_Europe': 0,
                'origin_Japan': 0,
                'origin_USA': 0,
                'sqr_horsepower': 0,
                'cube_horsepower': 0,
                'sqr_weight': 0,
                'cube_weight': 0,
                'prod_sqr_horsepower': 0,
                'prod_sqr_weight': 0
                }

        menu = '''
        [n]: Make a new predict
        [a]: Evaluate Model
        [q]: Quit
        => '''
        action = input(menu)

        if action == 'n':
            print('\n-------------------- NewPredict --------------------') # Preenchimento de um novo conjunto de features para previsão
            x_prev['cylinders'] = int(input("Cylinders: "))
            x_prev['displacement'] = float(input("Displacement: "))
            x_prev['horsepower'] = float(input("Horsepower: "))
            x_prev['weight'] = float(input("Weight: "))
            x_prev['model_year'] = int(input("Model Year: "))

            origin = int(input("Origin (1: EUA / 2: Europa / 3: Japan): "))
            if origin == 1:
                x_prev['origin_USA'] = 1
            elif origin == 2:
                x_prev['origin_Europe'] = 1
            elif origin == 3:
                x_prev['origin_Japan'] = 1

            x_prev['sqr_horsepower'] = x_prev['horsepower']**2
            x_prev['cube_horsepower'] = x_prev['horsepower']**3
            x_prev['sqr_weight'] = x_prev['weight']**2
            x_prev['cube_weight'] = x_prev['weight']**3
            x_prev['prod_sqr_horsepower'] = x_prev['horsepower'] * x_prev['sqr_horsepower']
            x_prev['prod_sqr_weight'] = x_prev['weight'] * x_prev['sqr_weight']

            x = np.array(list(x_prev.values()))
            x = (x - mu) / sigma # Normalizando as features com os mesmos valores da média e desvio padrão usados no treinamento

            print(f"\nO consumo previsto para este carro é em torno de {prediction(x, w, b):.2f} Km/L\n")
            print('--------------------------------------------------')
        elif action == 'a':
            print('\n--------------- Assessment Data ---------------')
            j_prints = [1, 50, 100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 2000]
            for i in range(len(J_history)): # Exibição do valor do custo nas iterações de n° acima
                index = i + 1
                if (index) in j_prints:
                    print(f'Iteration {index}: Cost(MSE): {J_history[index]:.2f}')
            print('\nCoefficients(W):', w) # Exibição dos valores dos coeficientes (w)
            print(f'Intercept(b): {b}\n') # Exibição do valor do intercepto (b)
            print('-----------------------------------------------')
        elif action == 'q':
            print('\n==================== Session finished ====================')
            break

main()