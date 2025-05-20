import pandas as pd
import numpy as np

def processing_data(df):
    df['origin'] = df['origin'].map({1: 'USA', 2:'Europe', 3:'Japan'}) # Alterando valores da coluna 'Origem'
    df = pd.get_dummies(df, columns=['origin'], dtype='int64') # Transformando valores da coluna 'Origem' em variáveis indicadoras

    df.drop(columns=['car_name', 'acceleration'], inplace=True) # Removendo colunas não utilizáveis

    df['weight_cyl'] = df['weight'] * df['cylinders'] # Adicionando novas features
    df['sqr_horsepower'] = df['horsepower']**2 # Adicionando novas features
    df['sqr_weight'] = df['weight']**2 # Adicionando novas features
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
    '''

    mu = np.mean(x, axis=0) # Média de cada feature
    sigma = np.std(x, axis=0) # Desvio padrão de cada feature

    x_norm = (x - mu) / sigma
    return x_norm

def compute_cost(x, y, w, b, lambda_):
    '''
    Função para calcular o custo
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

    # J_history = []
    # i_prints = [100, 500, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2700, 2800, 2900, 2999]

    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_func(x, y, w, b, lambda_)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db


        # --- Linhas de código para teste de performance ---
        # if i<100000:
        #     cost = cost_func(x, y, w, b, lambda_)
        #     J_history.append(cost)
        
        # if i in i_prints:
        #     print(f"Iteração {i}: Custo: {float(J_history[-1]):8.2f}")
        # --------------------------------------------------
    
    print("Training completed!\n")
    return w, b

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
    predict = np.dot(x, w) + b
    return predict

def training_model(df):
    '''
    Função que treina o modelo com os dados fornecidos
    Args:
        df (DataFrame (m. n)): DataFrame com todos os dados de treinamento. 'm' Conjuntos e 'n' Features

    Retorno:
        w_final (ndarray (n,)): Parâmetro 'w' após treinamento
        b_final (scalar)      : Parâmetro 'b' após treinamento
    '''
    df = processing_data(df) # Tratando os dados para serem utilizados no modelo

    y_train, x_train = conversion_data(df)
    x_train = zscore_normalization(x_train)

    w_init = np.zeros(x_train.shape[1])
    b_init = 0

    iterations = 2000
    alpha = 0.1
    lambda_ = 0.7

    w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, compute_gradient, compute_cost, alpha, lambda_, iterations)
    return w_final, b_final


def main():

    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'] # Definindo nomes das features
    df = pd.read_csv('data/auto-mpg.data', sep='\s+', names=columns) # Importando os dados de treinamento em um DataFrame

    w, b = training_model(df)
    
    # --- Linhas de código para teste de performance ---
    # for i in range(20):
    #     print(f"Previsão: {np.dot(x_train[i], w_final) + b_final:.2f}, Target: {y_train[i]:.2f}")
    
    # df.drop(columns=['mpg', 'km_l'], inplace=True)
    # for j in range(len(w_final)):
    #     print(f"{df.columns[j]}: w_final: {w_final[j]}")
    # --------------------------------------------------
    
    print('=============== CarFuel Regression Model ===============')
    while True:
        x_prev = {'cylinders': 0,
                'displacement': 0,
                'horsepower': 0,
                'weight': 0,
                'model_year': 0,
                'origin_Europe': 0,
                'origin_Japan': 0,
                'origin_USA': 0,
                'weight_cyl': 0,
                'sqr_horsepower': 0,
                'sqr_weight': 0}

        menu = '''
        [n]: Make a new predict
        [q]: Quit
        => '''
        action = input(menu)

        if action == 'n':
            print('\n-------------------- NewPredict --------------------')
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

            x_prev['weight_cyl'] = x_prev['weight'] * x_prev['cylinders']
            x_prev['sqr_horsepower'] = x_prev['horsepower']**2
            x_prev['sqr_weight'] = x_prev['weight']**2

            x = np.array(list(x_prev.values()))
            x = zscore_normalization(x)

            print(f"\nO consumo previsto para este carro é em torno de {prediction(x, w, b):.2f} Km/L\n")
            print('--------------------------------------------------')
        elif action == 'q':
            print('\n=============== Session finished ===============')
            break

main()