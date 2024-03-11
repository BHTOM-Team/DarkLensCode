import sys
# setting path
# adding Folder_2 to the system path
sys.path.insert(0, '../DarkLensCode')

import unittest

import src.mass_luminosity as ml


class TestMalkov(unittest.TestCase):
    def test_ms_star(self):
        masses = [1.0, 1.44, 1.93, 2.05, 5.4, 20.]
        mamajek_mags = [2.94, 2.82, 1.61, 1.36, -1.19, -4.16]
        mag_min = -4.16
        mag_max = 2.94
        for i in range(len(masses)):
            abs_mag = ml.get_abs_mag_g_ms_malkov(masses[i], mag_min, mag_max)
            message = "mass=%.2f M_sun: Calculated abs_mag (%.2f) is not equal to test value (%.2f)."%(masses[i], abs_mag, mamajek_mags[i])
            self.assertAlmostEqual(abs_mag, mamajek_mags[i], 2, message)


if __name__ == '__main__':
    unittest.main()
