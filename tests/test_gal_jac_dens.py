import sys
# setting path
# adding Folder_2 to the system path
sys.path.insert(0, '../DarkLensCode')

import unittest
import numpy as np
import yaml

import src.gal_jac_dens as gjd


class MyTestCase(unittest.TestCase):
    def test_proper_motions(self):
        gl = 23.205062229635374
        gb = 0.9257453322487849
        dist = 0.41600562213143993
        mu_l, mu_b, sig_mu_l, sig_mu_b = gjd.get_galactic_pm(gl, gb, dist)
        message_l = "Calculated mu_l is not equal to test value."
        message_b = "Calculated mu_b is not equal to test value."
        self.assertAlmostEqual(mu_l, -4.199467, 4, message_l)
        self.assertAlmostEqual(mu_b, -3.615723, 4, message_b)

    def test_disc_pm_prob(self):
        gl = 23.20506
        gb = 0.925745
        ra = 277.56025
        dec = -8.22021
        dist_lens = 1.16948
        dist_source = 17.93133
        pien, piee = 0.0218495, 0.1188562
        piE = np.sqrt(pien**2 + piee**2)
        mu_ra, mu_dec = -2.861862185544017, -5.447365851300708
        sig_mu_ra, sig_mu_dec = 0.022097435, 0.02856878
        pm_corr = -0.017446825
        murel = 5.3207584

        prob_disc = gjd.prob_mu_disc(murel, dist_lens, dist_source,
                     pien, piee, piE,
                     mu_ra, mu_dec, sig_mu_ra, sig_mu_dec, pm_corr,
                     ra, dec, gl, gb,
                     False)
        message_l = "Calculated north component of disc pm prob is not equal to test value."
        message_b = "Calculated east component of disc pm prob is not equal to test value."
        self.assertAlmostEqual(prob_disc[0], 0.2021899, 4, message_l)
        self.assertAlmostEqual(prob_disc[1], 0.1829690, 4, message_b)

    def test_bulge_pm_prob(self):
        gl = 23.20506
        gb = 0.925745
        ra = 277.56025
        dec = -8.22021
        dist_lens = 1.16948
        dist_source = 17.93133
        pien, piee = 0.0218495, 0.1188562
        piE = np.sqrt(pien ** 2 + piee ** 2)
        mu_ra, mu_dec = -2.861862185544017, -5.447365851300708
        sig_mu_ra, sig_mu_dec = 0.022097435, 0.02856878
        pm_corr = -0.017446825
        murel = 5.3207584

        prob_bulge = gjd.prob_mu_bulge(murel, dist_lens, dist_source,
                                     pien, piee, piE,
                                     mu_ra, mu_dec, sig_mu_ra, sig_mu_dec, pm_corr,
                                     ra, dec, gl, gb,
                                     False
                                     )
        message_l = "Calculated north component of bulge pm prob is not equal to test value."
        message_b = "Calculated east component of bulge pm prob is not equal to test value."
        self.assertAlmostEqual(prob_bulge[0], 0.0547401, 4, message_l)
        self.assertAlmostEqual(prob_bulge[1], 0.0553943, 4, message_b)

    def test_source_pm_disc(self):
        gl = 23.20506
        gb = 0.925745
        dist_source = 17.93133
        mu_sl, mu_sb, sig_mu_sl, sig_mu_sb = gjd.get_source_pm(gl, gb, dist_source)
        message_l = "Calculated proper motion l is not equal to test value."
        message_b = "Calculated proper motion b is not equal to test value."
        self.assertAlmostEqual(mu_sl, -4.94325449, 4, message_l)
        self.assertAlmostEqual(mu_sb, -0.07846711, 4, message_b)
        message_l = "Calculated proper motion error l is not equal to test value."
        message_b = "Calculated proper motion error b is not equal to test value."
        self.assertAlmostEqual(sig_mu_sl, 0.35296383, 4, message_l)
        self.assertAlmostEqual(sig_mu_sb, 0.23530922, 4, message_b)

    def test_source_pm_bulge(self):
        gl = 0.
        gb = 0.
        dist_source = 8.
        mu_sl, mu_sb, sig_mu_sl, sig_mu_sb = gjd.get_source_pm(gl, gb, dist_source)
        message_l = "Calculated proper motion l is not equal to test value."
        message_b = "Calculated proper motion b is not equal to test value."
        self.assertAlmostEqual(mu_sl, -6.12, 2, message_l)
        self.assertAlmostEqual(mu_sb, -0.19, 2, message_b)
        message_l = "Calculated proper motion error l is not equal to test value."
        message_b = "Calculated proper motion error b is not equal to test value."
        self.assertAlmostEqual(sig_mu_sl, 2.64, 2, message_l)
        self.assertAlmostEqual(sig_mu_sb, 2.64, 2, message_b)

    def test_broken_mass_function(self):
        with open('examples/example_2.yaml', 'r') as input_yaml:
            settings = yaml.safe_load(input_yaml)
        try:
            options = settings['options']
        except KeyError:
            options = {}

        mass_pows, mass_break = gjd.initialize_mass_fun(options['masspower'])
        mass_prob = gjd.get_mass_prob(1.0, mass_pows, mass_break)
        self.assertAlmostEqual(mass_prob, 1.0)  # add assertion here
        mass_prob = gjd.get_mass_prob(0.1, mass_pows, mass_break)
        self.assertAlmostEqual(mass_prob, 0.1 ** (-1.3), places=4)  # add assertion here
        mass_prob = gjd.get_mass_prob(0.01, mass_pows, mass_break)
        self.assertAlmostEqual(mass_prob, 0.01 ** (-0.3), places=4)  # add assertion here

    def test_simple_mass_function(self):
        with open('examples/example.yaml', 'r') as input_yaml:
            settings = yaml.safe_load(input_yaml)
        try:
            options = settings['options']
        except KeyError:
            options = {}

        mass_pows, mass_break = gjd.initialize_mass_fun(options['masspower'])
        mass_prob = gjd.get_mass_prob(100.0, mass_pows, mass_break)
        self.assertAlmostEqual(mass_prob, 100.0 ** (-1.75), places=4)  # add assertion here
        mass_prob = gjd.get_mass_prob(1.0, mass_pows, mass_break)
        self.assertAlmostEqual(mass_prob, 1.0)  # add assertion here
        mass_prob = gjd.get_mass_prob(0.1, mass_pows, mass_break)
        self.assertAlmostEqual(mass_prob, 0.1 ** (-1.75), places=4)  # add assertion here
        mass_prob = gjd.get_mass_prob(0.01, mass_pows, mass_break)
        self.assertAlmostEqual(mass_prob, 0.01 ** (-1.75), places=4)  # add assertion here

if __name__ == '__main__':
    unittest.main()
