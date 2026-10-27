import unittest
from app import app


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = app.test_client()
        cls.c.post('/train', json={'mode': 'test'})

    def test_01_train(self):
        self.assertTrue(self.c.post('/train', json={'mode': 'test'}).get_json()['trained'])

    def test_02_predict(self):
        r = self.c.post('/predict', json={'country': 'all', 'date': '2018-01-05', 'mode': 'test'})
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.get_json()['y_pred'], float)

    def test_03_predict_bad(self):
        self.assertEqual(self.c.post('/predict', json={'mode': 'test'}).status_code, 400)
        r = self.c.post('/predict', json={'country': 'atlantis', 'date': '2018-01-05', 'mode': 'test'})
        self.assertEqual(r.status_code, 400)

    def test_04_logs(self):
        test_logs = [f for f in self.c.get('/logs').get_json() if f.startswith('test-predict')]
        self.assertTrue(test_logs)
        self.assertEqual(self.c.get(f'/logs/{test_logs[0]}').status_code, 200)
