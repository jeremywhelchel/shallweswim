from parameterized import parameterized
import numpy as np
import pandas as pd
import unittest

from shallweswim import util


class TestData(unittest.TestCase):
    def test_pivot_year(self):
        df = pd.DataFrame(
            {"air_temp": 60, "water_temp": 50},
            index=pd.date_range("2011-06-01", "2023-03-01"),
        )
        got = util.PivotYear(df)

        pd.testing.assert_index_equal(
            pd.date_range("2020-01-01", "2020-12-31"), got.index
        )

        cols = ["air_temp", "water_temp"]
        years = pd.Series(range(2011, 2023 + 1), dtype="int32")
        pd.testing.assert_index_equal(
            pd.MultiIndex.from_product([cols, years], names=[None, "year"]), got.columns
        )
        self.assertTrue(got["air_temp"].isin([60, np.nan]).all().all())
        self.assertTrue(got["water_temp"].isin([50, np.nan]).all().all())

    @parameterized.expand(
        [
            ("fahrenheit", "celsius", "expected"),
            (32, 0, 0),
            (212, 100, 100),
            (98.6, 37, 37),
            (0, -17.77777777777778, -17.77777777777778),
        ]
    )
    def test_f2c(self, fahrenheit, celsius, expected):
        self.assertEqual(util.F2C(fahrenheit), celsius)


if __name__ == "__main__":
    unittest.main()
