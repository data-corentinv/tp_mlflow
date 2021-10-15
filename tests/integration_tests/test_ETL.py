import os
import unittest
import pandas as pd
from foodcast.settings import TEST_DATA_DIR # type: ignore
from foodcast.domain.transform import etl


class TestETL(unittest.TestCase):

    def test_etl(self) -> None:
        result = etl(TEST_DATA_DIR, 150, 151)
        expected = pd.read_csv(
            os.path.join(TEST_DATA_DIR, 'expected', 'load_expected.csv'),
            parse_dates=['order_date']
        )
        pd.testing.assert_frame_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
