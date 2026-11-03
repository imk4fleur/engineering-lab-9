import io
import unittest
import numpy as np
import pandas as pd
from app import app, analyze, demo_csv


class AppTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        app.config['TRAINED_MODEL'] = None

    def test_demo_analysis(self):
        csv = demo_csv(21000)
        data = pd.read_csv(io.StringIO(csv))
        self.assertIn('energy_kwh', data)
        self.assertTrue(((data.energy_kwh >= 80).astype(int) == data.high_usage).all())
        result = analyze(io.BytesIO(csv.encode()))
        self.assertEqual(result['rows'], 21000)
        self.assertAlmostEqual(result['means'][2], data.energy_kwh.mean(), places=8)
        self.assertEqual(sum(x['count'] for x in result['histogram']), 21000)
        self.assertEqual(sum(x['count'] for x in result['temperature_bins']), 21000)
        self.assertEqual(int(np.array(result['matrix']).sum()), result['test_rows'])
        response = self.client.post('/analyze', data={'file': (io.BytesIO(csv.encode()), 'demo.csv')})
        self.assertEqual(response.status_code, 200)
        for label in ('21000', 'Распределение потребления', 'Расход и температура', 'Матрица ошибок', 'Accuracy'):
            self.assertIn(label, response.get_data(as_text=True))

    def test_prediction_after_training(self):
        missing = self.client.post('/predict', json={'temperature': 0, 'occupants': 10, 'model_id': 'missing'})
        self.assertEqual(missing.status_code, 409)
        self.client.post('/analyze', data={'file': (io.BytesIO(demo_csv(5000).encode()), 'demo.csv')})
        state = app.config['TRAINED_MODEL']
        response = self.client.post('/predict', json={'temperature': -10, 'occupants': 80, 'model_id': state['id']})
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        expected = state['model'].predict_proba(state['scaler'].transform(pd.DataFrame([[-10, 80]], columns=['outside_temp', 'occupants'])))[0, 1]
        self.assertAlmostEqual(result['probability'], expected * 100, places=1)
        self.assertEqual(result['high_usage'], bool(expected >= .5))
        for bad in ({'temperature': 'nan', 'occupants': 10}, {'temperature': 2, 'occupants': -1}, {'temperature': 2, 'occupants': 1.5}, {'temperature': 2}):
            bad['model_id'] = state['id']
            self.assertEqual(self.client.post('/predict', json=bad).status_code, 400)
        out = self.client.post('/predict', json={'temperature': 90, 'occupants': 1, 'model_id': state['id']})
        self.assertTrue(out.get_json()['outside_range'])
        self.client.post('/analyze', data={'file': (io.BytesIO(demo_csv(5000, seed=7).encode()), 'new.csv')})
        self.assertEqual(self.client.post('/predict', json={'temperature': 2, 'occupants': 1, 'model_id': state['id']}).status_code, 409)

    def test_demo_energy_nonnegative(self):
        data = pd.read_csv(io.StringIO(demo_csv(100000, seed=1030042)))
        self.assertTrue((data.energy_kwh >= 0).all())

    def test_bad_schema(self):
        response = self.client.post('/analyze', data={'file': (io.BytesIO(b'outside_temp,occupants,high_usage\n1,2,0\n'), 'old.csv')})
        self.assertEqual(response.status_code, 400)
        self.assertIn('energy_kwh', response.get_data(as_text=True))

    def test_bad_label(self):
        data = b'outside_temp,occupants,energy_kwh,high_usage\n1,10,40,2\n'
        response = self.client.post('/analyze', data={'file': (io.BytesIO(data), 'bad.csv')})
        self.assertEqual(response.status_code, 400)
        self.assertIn('high_usage допустимы только 0 и 1', response.get_data(as_text=True))

    def test_negative_energy(self):
        data = b'outside_temp,occupants,energy_kwh,high_usage\n1,10,-1,0\n'
        response = self.client.post('/analyze', data={'file': (io.BytesIO(data), 'bad.csv')})
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
