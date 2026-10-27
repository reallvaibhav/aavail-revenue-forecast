import os, unittest
from model import model_train, model_load, model_predict, MODEL_DIR, VERSION


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_train(test=True)

    def test_01_train(self):
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, f'test-all-{VERSION}.joblib')))

    def test_02_load(self):
        self.assertTrue(hasattr(model_load(test=True)['all'], 'predict'))

    def test_03_predict(self):
        self.assertIsInstance(model_predict('all', '2018-01-05', test=True)['y_pred'], float)

    def test_04_bad_input(self):
        with self.assertRaises(ValueError):
            model_predict('atlantis', '2018-01-05', test=True)
        with self.assertRaises(ValueError):
            model_predict('all', '1990-01-01', test=True)
