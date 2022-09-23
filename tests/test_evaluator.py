import unittest
from pathlib import Path

from censyn.datasources import datasource
from censyn.evaluator import evaluator
from censyn.metrics import metrics
from censyn.results import result


class TestEvaluator(unittest.TestCase):

    def setUp(self):

        table_soc_cha_15_name = 'ACS_15_5YR_DP02_socCh.csv'
        table_soc_cha_16_name = 'ACS_16_5YR_DP02_socCh.csv'

        assets_path = Path(__file__).resolve().parent.parent / 'tests' / 'assets'

        table_1_path = str(assets_path / table_soc_cha_15_name)
        table_2_path = str(assets_path / table_soc_cha_16_name)

        csv_data_source_1 = datasource.DelimitedDataSource(path_to_file=table_1_path)
        csv_data_source_2 = datasource.DelimitedDataSource(path_to_file=table_2_path)

        self.df_1 = csv_data_source_1.to_dataframe()

        self.df_2 = csv_data_source_2.to_dataframe()

        metric_marginal_metric = metrics.MarginalMetric(sample_ratio=0.000001, name='TEST')
        self.evaluation_metrics = [metric_marginal_metric]

        self.evaluator = evaluator.Evaluator(self.df_1, self.df_2, self.evaluation_metrics)

    def test_evaluate(self) -> None:
        """
        Test:
            1) Make sure returns a dict, and that len(dict) == len(evaluation metrics)
            2) Raise exception on empty evaluation metrics
        """

        evaluation_results = self.evaluator.evaluate()

        self.assertIsInstance(evaluation_results, dict)
        for key, val in evaluation_results.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(val, result.Result)

        self.assertEqual(len(evaluation_results), len(self.evaluation_metrics))

        self.evaluation_metrics = []

        with self.assertRaises(ValueError) as cm:
            self.evaluator = evaluator.Evaluator(self.df_1, self.df_2, self.evaluation_metrics)
        err = cm.exception
        self.assertEqual(str(err), 'You must provide evaluation metrics to the Evaluator.')


if __name__ == '__main__':
    unittest.main()
