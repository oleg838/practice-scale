import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.signal import find_peaks
import joblib
import librosa
import numpy as np
from tqdm import tqdm
import os

def train_model(dataset_path):

    files = []

    file_list = [filename for filename in os.listdir(dataset_path) if filename.endswith(".wav")]

    for filename in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(dataset_path, filename)
        try:
            y, sr = librosa.load(file_path, sr=None)
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            peaks, _ = find_peaks(rms, height=np.mean(rms) + 0.5 * np.std(rms), distance=sr/(hop_length*2))
            min_distance = 0.4  
            current_group = []
            group_2 = []

            for i in range(len(peaks)-1):
                if (peaks[i] - peaks[i+1]) >= -30:
                    current_group.append(peaks[i])
            files.append([filename.split('.wav')[0], len(current_group)])
        except Exception as e:
            files.append([filename.split('.wav')[0], 0])
    files_df = pd.DataFrame(files, columns=['id', 'ring_count'])
    files_df['id'] = files_df['id'].astype(float)

    # Загрузка данных
    df = pd.read_csv(dataset_path+"/info.csv", sep=';')
    df = pd.get_dummies(df)
    df = df.astype(float)

    merged_df = df.merge(files_df, left_on='ID записи', right_on='id', how='left')
    df = merged_df.drop(['id'],axis=1)
    df.fillna(0)
    df['ring_count'] = df['ring_count'].astype(float)
    df = pd.get_dummies(df)
    df = df.astype(float)
    df.head()

    X = df.drop(columns=['Успешный результат'])
    y = df['Успешный результат']

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Параметры модели
    param = {
        'max_depth': 5,
        'learning_rate': 0.011,
        'n_estimators': 189,
        'subsample': 0.77,
        'colsample_bytree': 0.73,
        'gamma': 0.16,
        'min_child_weight': 5,
        'reg_alpha': 1,  # L1 регуляризация
        'reg_lambda': 1
    }

    # Создание и обучение модели
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)

    # Сохранение модели
    joblib.dump(model, 'model.joblib')

    # Оценка точности модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

def process_data(dataset_path):

    files = []

    file_list = [filename for filename in os.listdir(dataset_path) if filename.endswith(".wav")]

    for filename in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(dataset_path, filename)
        try:
            y, sr = librosa.load(file_path, sr=None)
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            peaks, _ = find_peaks(rms, height=np.mean(rms) + 0.5 * np.std(rms), distance=sr/(hop_length*2))
            min_distance = 0.4  
            current_group = []
            group_2 = []

            for i in range(len(peaks)-1):
                if (peaks[i] - peaks[i+1]) >= -30:
                    current_group.append(peaks[i])
            files.append([filename.split('.wav')[0], len(current_group)])
        except Exception as e:
            files.append([filename.split('.wav')[0], 0])
    files_df = pd.DataFrame(files, columns=['id', 'ring_count'])
    files_df['id'] = files_df['id'].astype(float)

    PATH_Name = "/info.csv"

    # Загрузка данных
    df = pd.read_csv(dataset_path+PATH_Name, sep=';')
    filenames = df['ID записи']
    merged_df = df.merge(files_df, left_on='ID записи', right_on='id', how='left')
    df = merged_df.drop(['id'],axis=1)
    df.fillna(0)
    df['ring_count'] = df['ring_count'].astype(float)
    df = pd.get_dummies(df)
    df = df.astype(float)
    df.head()
    X = df
    try:
        X = X.drop(columns=['Успешный результат'])
    except Exception as e:
        print("Так и должно быть",e)


    # Загрузка модели
    model = joblib.load('model.joblib')
    
    # Предсказание меток
    y_pred = model.predict(X)

    # Сохранение результатов в файл
    with open('result.txt', 'w') as f:
        for filename, pred in zip(filenames, y_pred):
            f.write(f'{filename}-{pred}\n')

def main():
    parser = argparse.ArgumentParser(description="Скрипт для обучения и обработки данных")
    parser.add_argument('--learn', action='store_true', help="Запуск режима обучения")
    parser.add_argument('--dataset', type=str, help="Путь к набору данных", required=True)

    args = parser.parse_args()

    if args.learn:
        train_model(args.dataset)
    else:
        process_data(args.dataset)

if __name__ == '__main__':
    main()
