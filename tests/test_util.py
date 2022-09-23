import unittest

import pandas as pd

from censyn.utils import StableFeatures


class UtilTest(unittest.TestCase):

    def test_stable_features(self) -> None:
        """Test for StableFeatures."""
        data_a = [0, 1, 0, 1, 2, 1, 0, 2, 1, 0, 1, 1]
        data_b = [0, 1, 1, 1, 2, 1, 0, 1, 1, 1, 0, 1]
        density_a = [data_a[i] / sum(data_a) for i in range(len(data_a))]
        density_b = [data_b[i] / sum(data_b) for i in range(len(data_b))]
        diff = [abs(density_a[i] - density_b[i]) for i in range(len(data_a))]
        arrays = [['a', 'b', 'c'], [1, 2], ['Y', 'Z']]

        index = pd.MultiIndex.from_product(arrays, names=['first', 'second', 'third'])
        test_df = pd.DataFrame({0: diff, 1: density_a, 2: density_b}, index=index)
        sf = StableFeatures(names=['first'])
        sf.add_scores(test_df)
        out_df = sf.scores()
        out_series = out_df.loc[:, 'Raw score']
        out_series.sort_index(inplace=True)
        test_values = [0.400000, 0.222222, 0.6666666]
        for i in range(len(test_values)):
            self.assertAlmostEqual(out_series[i], test_values[i], places=5)


if __name__ == '__main__':
    unittest.main()
