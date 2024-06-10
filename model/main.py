import json
import pickle
import os
import pandas as pd
import tqdm
import warnings
import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.utils import compute_sample_weight
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel


warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(debug=True)


class PredictionInput(BaseModel):
    data: list[dict]


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                    num_parts_to_read: int = 12, columns=None, verbose=False) -> pd.DataFrame:
    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                            if filename.startswith('train')])
    print(dataset_paths)

    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
        print('chunk_path', chunk_path)
        chunk = pd.read_parquet(chunk_path, columns=columns)
        res.append(chunk)

    return pd.concat(res).reset_index(drop=True)


def prepare_transactions_dataset(path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1,
                                 num_parts_total: int = 50,
                                 save_to_path=None, verbose: bool = False):
    preprocessed_frames = []

    for step in tqdm.tqdm_notebook(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                                   desc="Transforming transactions data"):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once,
                                                             verbose=verbose)

        if save_to_path:
            block_as_str = str(step)
            if len(block_as_str) == 1:
                block_as_str = '00' + block_as_str
            else:
                block_as_str = '0' + block_as_str
            transactions_frame.to_parquet(os.path.join(save_to_path, f'processed_chunk_{block_as_str}.parquet'))

        preprocessed_frames.append(transactions_frame)
    return pd.concat(preprocessed_frames)


def feature_engineering(df):
    df = df.copy()

    df['total_overdue_flags'] = (
            df['is_zero_loans5'] + df['is_zero_loans530'] +
            df['is_zero_loans3060'] + df['is_zero_loans6090'] +
            df['is_zero_loans90']
    )
    df['total_overdue_counts'] = (
            df['pre_loans5'] + df['pre_loans530'] +
            df['pre_loans3060'] + df['pre_loans6090'] +
            df['pre_loans90']
    )
    df['credit_history_len'] = df['pre_since_opened'] - df['pre_since_confirmed']
    df['close_flags'] = df['pclose_flag'] - df['fclose_flag']

    return df


def convert_data_types(df):
    df = df.copy()
    col = df.columns
    df[col] = df[col].apply(pd.to_numeric, downcast="integer")
    df[col] = df[col].apply(pd.to_numeric, downcast="float")

    return df


def create_dummies(df):
    df = df.copy()
    features = list(set(df.columns) - set(['id', 'rn', 'flag']))
    features.sort()
    dummies = pd.get_dummies(df[features], columns=features)
    df = pd.concat([df[['id', 'rn', 'flag']], dummies], axis=1)
    df = df[['id', 'rn', 'flag'] + sorted(dummies.columns.tolist())]
    return df


def aggregate(df):
    df = df.copy()
    agg_df = {f: 'sum' for f in set(df.columns) - set(['id', 'rn', 'flag'])}
    agg_df['rn'] = 'count'
    columns_order = ['id', 'flag'] + sorted(list(agg_df.keys()))
    df = df[columns_order]
    df = df.groupby(['id', 'flag']).agg(agg_df).astype('int').reset_index(drop=False)

    single_valued_columns = df.columns[df.nunique() == 1]
    df = df.drop(single_valued_columns, axis=1)

    return df


def scale_features(df):
    df = df.copy()
    continuous_features = [col for col in df.columns if col not in ['id', 'flag'] and df[col].nunique() > 2]

    # Apply StandardScaler
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    return df


def drop_columns(df, columns_to_drop):
    df = df.copy()
    df.drop(columns_to_drop, axis=1, inplace=True)
    return df


def prepare_data():
    return Pipeline([
        ('feature_engineering', FunctionTransformer(feature_engineering, validate=False)),
        ('convert_data_types', FunctionTransformer(convert_data_types, validate=False)),
        ('create_dummies', FunctionTransformer(create_dummies, validate=False)),
        ('aggregate', FunctionTransformer(aggregate, validate=False)),
        ('scale_features', FunctionTransformer(scale_features, validate=False)),
        ('drop_columns', FunctionTransformer(drop_columns, kw_args={'columns_to_drop': ['id']}, validate=False)),
        ('convert_data_types_again', FunctionTransformer(convert_data_types, validate=False))
    ])


@app.post("/fit")
def fit_model(path: str = Form(...)):

    logging.info(f"Чтение и обработка данных из {path}")
    data = prepare_transactions_dataset(path, num_parts_to_preprocess_at_once=2, num_parts_total=2, save_to_path=path)
    targets = pd.read_csv(os.path.join(path, 'train_target.csv'))
    df = data.merge(targets, on='id')
    logging.info("Подготовка данных завершена")

    logging.info("Инициализация XGBoost classifier")
    xgbc = XGBClassifier(learning_rate=0.2, n_estimators=400, max_depth=3)

    logging.info("Подготовка данных для тренировки модели")
    pipe = prepare_data()
    scaled_df = pipe.fit_transform(df)
    X = scaled_df.drop(columns=['flag'], axis=1)
    y = scaled_df['flag']
    X.columns = X.columns.str.replace("-", "x")

    logging.info("Подсчет весов")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)

    logging.info("Фиттинг модели")
    xgbc.fit(X, y, sample_weight=sample_weights)

    logging.info("Предсказания модели")
    y_pred = xgbc.predict(X)
    y_pred_proba = xgbc.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, y_pred_proba)
    report = classification_report(y, y_pred, output_dict=True)
    logging.info(f"Тренировка модели завершена с ROC AUC: {roc_auc}")

    model_path = os.path.join('model', 'xgbc_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(xgbc, f)

    first_row = X.iloc[0].to_dict()
    json_data = {"data": [first_row]}
    json_path = os.path.join('model', 'first_row_features.json')
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)
    logging.info(f"Признаки первой строки сохранены в {json_path}")

    return {
        'message': "Модель успешно обучена и сохранена",
        'roc_auc': roc_auc,
        'report': report
    }


@app.post("/prediction")
def predict(input_data: PredictionInput):
    logger.info("Начинаем обработку запроса на прогнозирование")

    logger.info("Полученные входные данные:")
    logger.info(input_data.data)

    model_path = os.path.join('model', 'xgbc_model.pkl')
    if not os.path.exists(model_path):
        logger.error("Модель не найдена по пути: %s", model_path)
        raise HTTPException(status_code=404, detail="Модель не найдена")

    logger.info("Загружаем модель из файла")
    with open(model_path, 'rb') as f:
        xgbc = pickle.load(f)

    expected_features = xgbc.get_booster().feature_names
    logger.info("Ожидаемые моделью признаки: %s", expected_features)

    logger.info("Создаем DataFrame из входных данных")

    df = pd.DataFrame(input_data.data)

    logger.info("Проверка DataFrame на пустоту")
    if df.empty:
        logger.error("Введенные данные пусты")
        raise HTTPException(status_code=400, detail="Введенные данные пусты")

    logger.info("Замена дефисов в названиях столбцов на 'x'")
    X = df.copy()
    X.columns = X.columns.str.replace("-", "x")

    missing_features = set(expected_features) - set(X.columns)
    if missing_features:
        logger.error("Отсутствуют необходимые признаки: %s", list(missing_features))
        raise HTTPException(status_code=400, detail=f"Отсутствуют необходимые признаки: {list(missing_features)}")

    logger.info("Выполнение прогнозирования с использованием модели")
    predictions = xgbc.predict(X).tolist()
    predictions_proba = xgbc.predict_proba(X)[:, 1].tolist()

    logger.info("Прогнозирование завершено успешно")
    return {
        'predictions': predictions,
        'predictions_proba': predictions_proba
    }


if __name__ == '__main__':
    import uvicorn
    logger.info("Запуск приложения")
    uvicorn.run(app, host='127.0.0.3', port=8000)
