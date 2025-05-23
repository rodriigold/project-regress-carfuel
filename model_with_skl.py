import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def processing_data():
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    df = pd.read_csv('data/auto-mpg.data', sep='\s+', names=columns)

    df.drop(columns=['car_name', 'acceleration', 'origin'], inplace=True)

    df['sqr_weight'] = df['weight']**2
    return df

def conversion_data(df):
    df['km_l'] = df['mpg'] * 0.425144

    y = df['km_l'].values
    x = df.drop(columns=['mpg', 'km_l']).values
    return x, y

def main():
    # ========= Tratamento/Processamento dos dados =========
    df = processing_data()
    X, y = conversion_data(df)

    # -------- Definição das Features e dos Targets --------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------- Normalização --------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # ========= Treinamento =========
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    # ========= Avaliação =========
    cost = mean_squared_error(y_test, y_test_pred)
    r2test = r2_score(y_test, y_test_pred)

    r2train = r2_score(y_train, y_train_pred)

    print(f'Erro quadrático médio (Custo): {cost:.2f}')
    print(f'Coeficiente de determinação nos treinos (R^2): {r2train:.2f}')
    print(f'Coeficiente de determinação das teste (R^2): {r2test:.2f}')

    print('Coeficientes(Ws):', model.coef_)
    print('Intercept(b):', model.intercept_)


    # ========= Aplicação =========
    print('=============== CarFuel Regression Model (With Sklearn) ===============')
    while True:
        x_prev = {'cylinders': 0,
                'displacement': 0,
                'horsepower': 0,
                'weight': 0,
                'model_year': 0,
                'origin_Europe': 0,
                'origin_Japan': 0,
                'origin_USA': 0,
                'sqr_weight': 0}

        menu = '''
        [n]: Make a new predict
        [q]: Quit
        => '''
        action = input(menu)

        if action == 'n':
            print('\n-------------------- NewPredict --------------------')
            x_prev['cylinders'] = float(input("Cylinders: "))
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

            x_prev['sqr_weight'] = x_prev['weight']**2

            new_x_test = np.array([list(x_prev.values())])
            new_x_test_norm = scaler.transform(new_x_test)
            new_y_test = model.predict(new_x_test_norm)

            print(f"\nO consumo previsto para este carro é em torno de {new_y_test[0]:.2f} Km/L\n")
            print('--------------------------------------------------')
        elif action == 'q':
            print('\n=============== Session finished ===============')
            break

main()