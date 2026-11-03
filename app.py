import tempfile
import uuid

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024
app.config['TRAINED_MODEL'] = None
FEATURES = ['outside_temp', 'occupants']
COLUMNS = FEATURES + ['energy_kwh', 'high_usage']


def demo_csv(size=5000, seed=42):
    rng = np.random.default_rng(seed)
    temperature = rng.uniform(-20, 35, size).round(1)
    occupants = rng.integers(0, 101, size)
    # Синтетическое суточное потребление, кВт·ч; высокий расход — от 80 кВт·ч.
    energy = np.maximum(0, 20 + 0.6 * occupants + 1.2 * np.abs(temperature - 20) + rng.normal(0, 6, size)).round(2)
    return pd.DataFrame({'outside_temp': temperature, 'occupants': occupants,
                         'energy_kwh': energy, 'high_usage': (energy >= 80).astype(int)}).to_csv(index=False)


def chunks(file):
    file.seek(0)
    for frame in pd.read_csv(file, chunksize=10000):
        if not set(COLUMNS).issubset(frame.columns):
            raise ValueError('CSV должен содержать колонки outside_temp, occupants, energy_kwh и high_usage.')
        frame = frame[COLUMNS].apply(pd.to_numeric, errors='coerce')
        if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
            raise ValueError('Все значения должны быть заполненными конечными числами.')
        if (frame['energy_kwh'] < 0).any() or (frame['occupants'] < 0).any():
            raise ValueError('Расход и число людей не могут быть отрицательными.')
        if not frame['high_usage'].isin([0, 1]).all():
            raise ValueError('В колонке high_usage допустимы только 0 и 1.')
        yield frame


def analyze(file):
    scaler = StandardScaler()
    rows = positives = train_rows = test_rows = 0
    sums = np.zeros(3)
    minima = np.full(3, np.inf)
    maxima = np.full(3, -np.inf)
    train_classes = set()
    measures = FEATURES + ['energy_kwh']
    # Индексы сохраняются между порциями: каждая пятая строка — тестовая.
    for frame in chunks(file):
        test = frame.index % 5 == 0
        train = frame.loc[~test]
        rows += len(frame)
        positives += int(frame.high_usage.sum())
        sums += frame[measures].sum().to_numpy()
        minima = np.minimum(minima, frame[measures].min().to_numpy())
        maxima = np.maximum(maxima, frame[measures].max().to_numpy())
        train_rows += len(train)
        test_rows += int(test.sum())
        train_classes.update(train.high_usage.unique())
        if len(train):
            scaler.partial_fit(train[FEATURES])
    if rows < 10 or train_classes != {0, 1}:
        raise ValueError('Нужно минимум 10 строк и оба класса high_usage в обучающей части.')

    # Одинаковые границы для всех порций: иначе столбцы разных диапазонов нельзя складывать.
    energy_edges = np.linspace(minima[2], maxima[2] if maxima[2] > minima[2] else minima[2] + 1, 11)
    temp_edges = np.linspace(minima[0], maxima[0] if maxima[0] > minima[0] else minima[0] + 1, 11)
    energy_counts = np.zeros(10, dtype=np.int64)
    temp_counts = np.zeros(10, dtype=np.int64)
    temp_energy = np.zeros(10)
    model = SGDClassifier(loss='log_loss', random_state=42)
    for frame in chunks(file):
        energy_counts += np.histogram(frame.energy_kwh, bins=energy_edges)[0]
        temp_counts += np.histogram(frame.outside_temp, bins=temp_edges)[0]
        temp_energy += np.histogram(frame.outside_temp, bins=temp_edges, weights=frame.energy_kwh)[0]
        train = frame.loc[frame.index % 5 != 0]
        if len(train):
            model.partial_fit(scaler.transform(train[FEATURES]), train.high_usage, classes=np.array([0, 1]))

    matrix = np.zeros((2, 2), dtype=np.int64)
    for frame in chunks(file):
        test = frame.loc[frame.index % 5 == 0]
        if len(test):
            predicted = model.predict(scaler.transform(test[FEATURES])).astype(int)
            matrix += np.bincount(test.high_usage.to_numpy(dtype=int) * 2 + predicted, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = matrix.ravel()
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None
    histogram = [dict(label=f'{energy_edges[i]:.1f}–{energy_edges[i+1]:.1f}', count=int(energy_counts[i])) for i in range(10)]
    temperature_bins = [dict(label=f'{temp_edges[i]:.1f}–{temp_edges[i+1]:.1f}', count=int(temp_counts[i]),
                             mean=float(temp_energy[i] / temp_counts[i]) if temp_counts[i] else None) for i in range(10)]
    return dict(rows=rows, positives=positives, negative=rows-positives,
                percent=round(100 * positives / rows, 1), means=sums / rows,
                minimum=minima.tolist(), maximum=maxima.tolist(), histogram=histogram,
                temperature_bins=temperature_bins, matrix=matrix.tolist(), precision=precision, recall=recall, f1=f1,
                _trained=(scaler, model),
                train_rows=train_rows, test_rows=test_rows, accuracy=round(100 * (tn + tp) / test_rows, 1))


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/demo.csv')
def demo():
    return Response(demo_csv(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=demo.csv'})


@app.post('/analyze')
def analyze_upload():
    upload = request.files.get('file')
    if not upload or not upload.filename:
        return render_template('index.html', error='Выберите CSV-файл.'), 400
    try:
        with tempfile.TemporaryFile() as file:
            upload.save(file)
            result = analyze(file)
        scaler, model = result.pop('_trained')
        model_id = uuid.uuid4().hex
        # shortcut: одна модель на процесс, для нескольких пользователей нужен отдельный реестр моделей.
        app.config['TRAINED_MODEL'] = {'id': model_id, 'scaler': scaler, 'model': model,
                                       'minimum': result['minimum'][:2], 'maximum': result['maximum'][:2]}
        return render_template('index.html', result=result, model_id=model_id)
    except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return render_template('index.html', error=f'Не удалось обработать файл: {error}'), 400


@app.post('/predict')
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='Введите температуру и число людей.'), 400
    state = app.config['TRAINED_MODEL']
    if not state or data.get('model_id') != state['id']:
        return jsonify(error='Модель недоступна или заменена. Загрузите CSV и обучите модель заново.'), 409
    try:
        if isinstance(data.get('temperature'), bool) or isinstance(data.get('occupants'), bool):
            raise ValueError
        temperature = float(data['temperature'])
        occupants = float(data['occupants'])
        if not np.isfinite([temperature, occupants]).all() or not -100 <= temperature <= 100 or not 0 <= occupants <= 1000000 or not occupants.is_integer():
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        return jsonify(error='Укажите температуру от −100 до 100 °C и целое число людей от 0 до 1 000 000.'), 400
    features = pd.DataFrame([[temperature, occupants]], columns=FEATURES)
    probability = float(state['model'].predict_proba(state['scaler'].transform(features))[0, 1])
    outside_range = any(value < low or value > high for value, low, high in zip([temperature, occupants], state['minimum'], state['maximum']))
    return jsonify(probability=round(probability * 100, 1), high_usage=probability >= 0.5, outside_range=outside_range)


@app.errorhandler(413)
def too_large(error):
    return render_template('index.html', error='Файл превышает ограничение 256 МБ.'), 413


if __name__ == '__main__':
    app.run(debug=False)
