"""Project: Phi_K - correlation coefficient package

Created: 2018/11/13

Description:
    Collection of helper functions to get fixtures, i.e. for test data.
    These are mostly used by the (integration) tests and example notebooks.

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

import unittest
import unittest.mock as mock
import pytest


@pytest.mark.filterwarnings("ignore:Using or importing the ABCs from")
class PhiKTest(unittest.TestCase):
    """Tests for calculation of Phi_K"""

    def test_phik_calculation(self):
        """Test the calculation of Phi_K"""

        import numpy as np
        from phik import bivariate

        chi2 = bivariate.chi2_from_phik(0.5, 1000, nx=10, ny=10)
        self.assertTrue( np.isclose(chi2, 271.16068979654125, 1e-6) )

        phik = bivariate.phik_from_chi2(chi2, 1000, 10, 10)
        self.assertTrue( np.isclose(phik, 0.5, 1e-6) )

    def test_phik_matrix(self):
        """Test the calculation of Phi_K"""
        
        import numpy as np
        import pandas as pd
        import phik
        from phik import resources

        # open fake car insurance data
        df = pd.read_csv( resources.fixture('fake_insurance_data.csv.gz') )

        # get the phi_k correlation matrix between all variables
        interval_cols = ['driver_age', 'mileage']
        phik_corr = df.phik_matrix(interval_cols=interval_cols)

        self.assertTrue(np.isclose(phik_corr.values[1,0], 0.5904561614620166))
        self.assertTrue(np.isclose(phik_corr.values[2,4], 0.768588987856336))

    def test_global_phik(self):
        """Test the calculation of global Phi_K values"""

        import numpy as np
        import pandas as pd
        import phik
        from phik import resources

        # open fake car insurance data
        df = pd.read_csv( resources.fixture('fake_insurance_data.csv.gz') )

        # get the global phi_k values 
        interval_cols = ['driver_age', 'mileage']
        gk = df.global_phik(interval_cols=interval_cols)

        self.assertTrue(np.isclose(gk[0][0][0], 0.6057528003711345))
        self.assertTrue(np.isclose(gk[0][4][0], 0.768588987856336))

    def test_significance_matrix(self):
        """Test significance calculation"""

        import numpy as np
        import pandas as pd
        import phik
        from phik import resources

        # open fake car insurance data
        df = pd.read_csv( resources.fixture('fake_insurance_data.csv.gz') )

        # get significances
        interval_cols = ['driver_age', 'mileage']
        sm = df.significance_matrix(interval_cols=interval_cols, significance_method='asymptotic')

        self.assertTrue(np.isclose(sm.values[1,0], 37.66184429195198))
        self.assertTrue(np.isclose(sm.values[2,4], 49.3323049685695))

    def test_hist2d(self):
        """Test the calculation of global Phi_K values"""

        import numpy as np
        import pandas as pd
        import phik
        from phik import resources

        # open fake car insurance data
        df = pd.read_csv( resources.fixture('fake_insurance_data.csv.gz') )

        # create contingency matrix
        cols = ['mileage','car_size']
        interval_cols = ['mileage']
        h2d = df[cols].hist2d(interval_cols=interval_cols)

        self.assertEqual(h2d.values[1,1], 10)
        self.assertEqual(h2d.values[5,5], 217)

    def test_outlier_significance_matrix(self):
        """Test the calculation of outlier significances"""

        import numpy as np
        import pandas as pd
        import phik
        from phik import resources

        # open fake car insurance data
        df = pd.read_csv( resources.fixture('fake_insurance_data.csv.gz') )

        # calculate outlier significances
        cols = ['mileage','car_size']
        interval_cols = ['mileage']
        om = df[cols].outlier_significance_matrix(interval_cols=interval_cols)

        self.assertTrue(np.isclose(om.values[0,1], 21.483476494343552))
        self.assertTrue(np.isclose(om.values[2,4], -1.246784034214704))
