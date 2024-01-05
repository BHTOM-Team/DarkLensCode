import sys
# setting path
# adding Folder_2 to the system path
sys.path.insert(0, '../DarkLensCode')

import unittest
import packages.gal_jac_dens as gjd


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

    # def test_mass_function(self):
    #     with open('examples/example.yaml', 'r') as input_yaml:
    #         settings = yaml.safe_load(input_yaml)
    #     try:
    #         options = settings['options']
    #     except KeyError:
    #         options = {}
    #
    #     mass_pows, mass_break = initialize_mass_fun(options['masspower'])
    #     mass_prob = get_mass_prob(1.0, mass_pows, mass_break)
    #     self.assertAlmostEqual(mass_prob, 1.0)  # add assertion here
    #     mass_prob = get_mass_prob(0.1, mass_pows, mass_break)
    #     self.assertAlmostEqual(mass_prob, 0.1 ** (-1.3), places=4)  # add assertion here
    #     mass_prob = get_mass_prob(0.01, mass_pows, mass_break)
    #     self.assertAlmostEqual(mass_prob, 0.01 ** (-0.3), places=4)  # add assertion here

if __name__ == '__main__':
    unittest.main()
