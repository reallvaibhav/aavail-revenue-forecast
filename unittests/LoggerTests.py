import csv, os, unittest
from logger import update_train_log, update_predict_log


def last_row(path):
    with open(path) as f:
        return list(csv.DictReader(f))[-1]


class LoggerTest(unittest.TestCase):
    def test_01_train_log(self):
        path = update_train_log('unittest', '2018-01-01:2018-12-31', {'rf': 1.0}, 'rf', 0.1, '0.1', test=True)
        self.assertTrue(os.path.basename(path).startswith('test-'))
        self.assertEqual(last_row(path)['tag'], 'unittest')

    def test_02_predict_log(self):
        path = update_predict_log('unittest', '2018-01-05', 123.4, 0.1, '0.1', test=True)
        self.assertTrue(os.path.basename(path).startswith('test-'))
        self.assertEqual(float(last_row(path)['y_pred']), 123.4)
