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
import pytest

import pandas as pd
import numpy as np
from phik import resources, bivariate
from phik.simulation import sim_2d_data_patefield
from phik.binning import auto_bin_data, bin_data
from phik.phik import phik_observed_vs_expected_from_rebinned_df, phik_from_hist2d
from phik.statistics import get_dependent_frequency_estimates


@pytest.mark.filterwarnings("ignore:Using or importing the ABCs from")
class PhiKTest(unittest.TestCase):
    """Tests for calculation of Phi_K"""

    def test_phik_calculation(self):
        """Test the calculation of Phi_K"""

        chi2 = bivariate.chi2_from_phik(0.5, 1000, nx=10, ny=10)
        self.assertTrue(np.isclose(chi2, 271.16068979654125, 1e-6))

        phik = bivariate.phik_from_chi2(chi2, 1000, 10, 10)
        self.assertTrue(np.isclose(phik, 0.5, 1e-6))

    def test_phik_from_hist2d(self):
        """Test the calculation of Phi_K value from hist2d"""

        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))

        # create contingency matrix
        cols = ["mileage", "car_size"]
        interval_cols = ["mileage"]
        observed = df[cols].hist2d(interval_cols=interval_cols)

        phik_value = phik_from_hist2d(observed)
        self.assertAlmostEqual(phik_value, 0.7685888294891855, places=3)

    def test_phik_observed_vs_expected_from_hist2d(self):
        """Test the calculation of Phi_K value from hist2d"""

        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))

        # create contingency matrix
        cols = ["mileage", "car_size"]
        interval_cols = ["mileage"]

        observed = df[cols].hist2d(interval_cols=interval_cols).values
        expected = get_dependent_frequency_estimates(observed)

        phik_value = phik_from_hist2d(observed=observed, expected=expected)
        self.assertAlmostEqual(phik_value, 0.7685888294891855, places=3)

    def test_phik_matrix(self):
        """Test the calculation of Phi_K"""
        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))
        cols = list(df.columns)

        # get the phi_k correlation matrix between all variables
        interval_cols = ["driver_age", "mileage"]
        phik_corr = df.phik_matrix(interval_cols=interval_cols)

        self.assertAlmostEqual(
            phik_corr.values[cols.index("car_color"), cols.index("area")],
            0.5904561614620166,
            places=3,
        )
        self.assertAlmostEqual(
            phik_corr.values[cols.index("area"), cols.index("car_color")],
            0.5904561614620166,
            places=3,
        )
        self.assertAlmostEqual(
            phik_corr.values[cols.index("mileage"), cols.index("car_size")],
            0.768588987856336,
            places=3,
        )
        self.assertAlmostEqual(
            phik_corr.values[cols.index("car_size"), cols.index("mileage")],
            0.768588987856336,
            places=3,
        )

    def test_phik_matrix_observed_vs_expected(self):
        """Test the calculation of Phi_K"""
        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))
        cols = list(df.columns)

        # get the phi_k correlation matrix between all variables
        binned_df, _ = auto_bin_data(df)
        phik_corr = phik_observed_vs_expected_from_rebinned_df(binned_df, binned_df)

        self.assertTrue(
            np.isclose(
                phik_corr.values[cols.index("car_color"), cols.index("area")], 0.0
            )
        )
        self.assertTrue(
            np.isclose(
                phik_corr.values[cols.index("area"), cols.index("car_color")], 0.0
            )
        )
        self.assertTrue(
            np.isclose(
                phik_corr.values[cols.index("mileage"), cols.index("car_size")], 0.0
            )
        )
        self.assertTrue(
            np.isclose(
                phik_corr.values[cols.index("car_size"), cols.index("mileage")], 0.0
            )
        )
        self.assertTrue(
            np.isclose(
                phik_corr.values[cols.index("car_size"), cols.index("car_size")], 1.0
            )
        )

    def test_global_phik(self):
        """Test the calculation of global Phi_K values"""

        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))

        # get the global phi_k values
        interval_cols = ["driver_age", "mileage"]
        gk = df.global_phik(interval_cols=interval_cols)

        area = (np.where(gk[1] == "area"))[0][0]
        car_size = (np.where(gk[1] == "car_size"))[0][0]
        mileage = (np.where(gk[1] == "mileage"))[0][0]

        self.assertAlmostEqual(gk[0][area][0], 0.6057528003711345, places=3)
        self.assertAlmostEqual(gk[0][car_size][0], 0.76858883, places=3)
        self.assertAlmostEqual(gk[0][mileage][0], 0.768588987856336, places=3)

    def test_significance_matrix_asymptotic(self):
        """Test significance calculation"""

        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))
        cols = list(df.columns)
        # get significances
        interval_cols = ["driver_age", "mileage"]
        sm = df.significance_matrix(
            interval_cols=interval_cols, significance_method="asymptotic"
        )

        self.assertTrue(
            np.isclose(
                sm.values[cols.index("car_color"), cols.index("area")],
                37.66184429195198,
            )
        )
        self.assertTrue(
            np.isclose(
                sm.values[cols.index("area"), cols.index("car_color")],
                37.66184429195198,
            )
        )
        self.assertTrue(
            np.isclose(
                sm.values[cols.index("mileage"), cols.index("car_size")],
                49.3323049685695,
            )
        )
        self.assertTrue(
            np.isclose(
                sm.values[cols.index("car_size"), cols.index("mileage")],
                49.3323049685695,
            )
        )

    def test_significance_matrix_hybrid(self):
        """Test significance calculation"""

        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))
        cols = list(df.columns)
        # get significances
        interval_cols = ["driver_age", "mileage"]
        sm = df.significance_matrix(
            interval_cols=interval_cols, significance_method="hybrid"
        )

        self.assertTrue(
            np.isclose(
                sm.values[cols.index("car_color"), cols.index("area")],
                37.63086023595297,
                atol=10e-2,
            )
        )
        self.assertTrue(
            np.isclose(
                sm.values[cols.index("area"), cols.index("car_color")],
                37.63086023595297,
                atol=10e-2,
            )
        )
        self.assertTrue(
            np.isclose(
                sm.values[cols.index("mileage"), cols.index("car_size")],
                49.28345609465683,
                atol=10e-2,
            )
        )
        self.assertTrue(
            np.isclose(
                sm.values[cols.index("car_size"), cols.index("mileage")],
                49.28345609465683,
                atol=10e-2,
            )
        )

    def test_significance_matrix_mc(self):
        """Test significance calculation"""

        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))
        cols = list(df.columns)
        # get significances
        interval_cols = ["driver_age", "mileage"]
        sm = df.significance_matrix(
            interval_cols=interval_cols, significance_method="MC"
        )

        self.assertTrue(
            np.isclose(sm.values[cols.index("car_color"), cols.index("area")], np.inf)
        )
        self.assertTrue(
            np.isclose(sm.values[cols.index("area"), cols.index("car_color")], np.inf)
        )
        self.assertTrue(
            np.isclose(sm.values[cols.index("mileage"), cols.index("car_size")], np.inf)
        )
        self.assertTrue(
            np.isclose(sm.values[cols.index("car_size"), cols.index("mileage")], np.inf)
        )

    def test_hist2d(self):
        """Test the calculation of global Phi_K values"""

        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))

        # create contingency matrix
        cols = ["mileage", "car_size"]
        interval_cols = ["mileage"]
        h2d = df[cols].hist2d(interval_cols=interval_cols)

        self.assertEqual(h2d.values[1, 1], 10)
        self.assertEqual(h2d.values[5, 5], 217)

    def test_hist2d_array(self):
        """Test the calculation of global Phi_K values"""

        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))

        # create contingency matrix
        interval_cols = ["mileage"]
        h2d = df["mileage"].hist2d(df["car_size"], interval_cols=interval_cols)
        self.assertEqual(h2d.values[1, 1], 10)
        self.assertEqual(h2d.values[5, 5], 217)

    def test_outlier_significance_matrix(self):
        """Test the calculation of outlier significances"""

        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))

        # calculate outlier significances
        cols = ["mileage", "car_size"]
        interval_cols = ["mileage"]
        om = df[cols].outlier_significance_matrix(interval_cols=interval_cols)

        self.assertTrue(np.isclose(om.values[0, 1], 21.483476494343552))
        self.assertTrue(np.isclose(om.values[2, 4], -1.246784034214704))

    def test_outlier_significance_matrices(self):
        """Test the calculation of outlier significances"""

        # open fake car insurance data
        df = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))

        # calculate outlier significances
        interval_cols = ["mileage", "driver_age"]
        om = df.outlier_significance_matrices(interval_cols=interval_cols)

        self.assertTrue(isinstance(om, dict))

    def test_simulation_2d_patefield(self):
        """Test simulation code using patefield algorithm."""
        og_state = np.random.get_state()
        np.random.seed(42)
        sample = np.random.randint(1, 200, (50, 2))

        # call test function
        res = sim_2d_data_patefield(sample, seed=42).T
        np.random.set_state(og_state)
        mean0, mean1 = res.mean(1)
        self.assertTrue(np.isclose(mean0, 105.46))
        self.assertTrue(np.isclose(mean1, 91.18))

    def test_binning_bin_data_bins_tyes(self):
        # Non regression test
        # https://github.com/KaveIO/PhiK/issues/28
        df = pd.DataFrame({"x": np.random.randn(10)})
        bins_int = np.arange(5, 11, 1)
        bins_float = np.arange(5, 11, 1.0)
        bins_dict_int = {"x": np.uint8(10)}
        bins_dict_float = {"x": np.float32(10.3)}

        for bins in bins_int:
            bin_data(df, cols=["x"], bins=bins)

        for bins in bins_float:
            bin_data(df, cols=["x"], bins=bins)

        bin_data(df, cols=["x"], bins=bins_dict_int)
        bin_data(df, cols=["x"], bins=bins_dict_float)
