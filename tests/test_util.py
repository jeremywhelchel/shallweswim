import numpy as np
import pandas as pd
import unittest
import pytest

from shallweswim import util


import datetime
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

    def test_now(self):
        now = util.Now()
        self.assertIsInstance(now, datetime.datetime)
        self.assertIsNotNone(now.tzinfo)
        self.assertEqual(now.tzinfo.zone, "US/Eastern")

    @pytest.mark.parametrize("fahrenheit, celsius", [(32, 0), (212, 100), (68, 20), (-40, -40)])
    def test_f2c(self, fahrenheit, celsius):
        self.assertAlmostEqual(util.F2C(fahrenheit), celsius, places=1)

if __name__ == "__main__":
    unittest.main()
