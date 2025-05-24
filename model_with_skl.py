import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def processing_data():
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'] # Definindo nomes das features
    df = pd.read_csv('data/auto-mpg.data', sep=r'\s+', names=columns) # Importando os dados de treinamento em um DataFrame

    df.drop(columns=['car_name', 'acceleration', 'origin'], inplace=True) # Removendo colunas não utilizáveis
    return df

def conversion_data(df):
    df['km_l'] = df['mpg'] * 0.425144 # Convertendo valores do dataset de Milhas por galão para Quilômetros por Litro (Km/L)

    y = df['km_l'].values # Referenciando os valores dos targets
    x = df.drop(columns=['mpg', 'km_l']).values # Referenciando os valores das features
    return x, y

def main():
    # ========= Tratamento/Processamento dos dados =========
    df = processing_data() # A variável 'df' recebe um DataFrame com os dados já tratados e prontos pro treinamento
    X, y = conversion_data(df) # X e y recebem os valores referentes das features e targets, respectivamente, vindos do DataFrame

    # -------- Definição das Features e dos Targets --------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 'train_test_split' é um método do sklearn que separa uma parte das features e targets para o treinamento e outra parte para testar o modelo posteriormente com dados desconhecidos


    # -------- Pré-Processamento das Features --------
    scaler = StandardScaler() # Objeto responsável pela normalização das features
    poly = PolynomialFeatures(3)  # Objeto responsável pela criação de features polinomiais
    
    X_train = scaler.fit_transform(X_train) # Normalizando as features de treino
    X_test = scaler.transform(X_test) # Normalizando as features de teste, com os mesmos valores da primeira normalização

    X_train = poly.fit_transform(X_train) # Adicionando features polinomiais ao conjunto de treino
    X_test = poly.transform(X_test) # Adicionando features polinomiais ao conjunto de teste


    # ========= Treinamento =========
    model = Ridge(alpha=0.7) # Objeto responsável pela Regressão linear (Regularizada pelo tipo L2). Alpha representa a taxa de regularização
    model.fit(X_train, y_train) # Treinamento do modelo 

    y_train_pred = model.predict(X_train) # Previsão, após treinamento, com base nas features de treinamento
    y_test_pred = model.predict(X_test) # Previsão, após treinamento, com base nas features de teste 


    # ========= Aplicação =========
    print('=============== CarFuel Regression Model (With Sklearn) ===============')
    while True:
        x_prev = {
                'cylinders': 0,
                'displacement': 0,
                'horsepower': 0,
                'weight': 0,
                'model_year': 0,
                }

        menu = '''
        [n]: Make a new predict
        [a]: Evaluate Model
        [q]: Quit
        => '''
        action = input(menu)

        if action == 'n': 
            print('\n-------------------- NewPredict --------------------') # Preenchimento de um novo conjunto de features para previsão
            x_prev['cylinders'] = float(input("Cylinders: "))
            x_prev['displacement'] = float(input("Displacement: "))
            x_prev['horsepower'] = float(input("Horsepower: "))
            x_prev['weight'] = float(input("Weight: "))
            x_prev['model_year'] = int(input("Model Year: "))

            new_x_test = np.array([list(x_prev.values())])
            new_x_test_norm = scaler.transform(new_x_test) # Normalizando as features com os mesmos valores do treinamento
            new_x_test_poly = poly.transform(new_x_test_norm) # Adicionando as features polinomiais
            new_y_test = model.predict(new_x_test_poly) # Nova previsão

            print(f"\nO consumo previsto para este carro é em torno de {new_y_test[0]:.2f} Km/L\n")
            print('--------------------------------------------------')
        elif action == 'a':
            print('\n--------------- Assessment Data ---------------')
            cost = mean_squared_error(y_test, y_test_pred) # Valor do custo final (MSE)
            r2test = r2_score(y_test, y_test_pred) # Coeficiente de determinação das previsões de teste

            r2train = r2_score(y_train, y_train_pred) # Coeficiente de determinação das previsões de treinamento

            print(f'Final Cost(MSE): {cost:.2f}')
            print(f'Coeficiente de determinação nos treinos (R^2): {r2train:.2f}')
            print(f'Coeficiente de determinação das teste (R^2): {r2test:.2f}')

            print('Coefficients(Ws):', model.coef_)
            print('Intercept(b):', model.intercept_)
            print('-----------------------------------------------')
        elif action == 'q':
            print('\n=============== Session finished ===============')
            break

main()