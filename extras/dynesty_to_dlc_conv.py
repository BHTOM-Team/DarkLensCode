import pickle
import getopt
import sys
import numpy as np
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

def main(): 
	outname = "outputfile.npy"

	try:
		opts, args = getopt.getopt(sys.argv[1:], "f:l:", ["filename=", "outname="])
	except getopt.GetoptError as err:
		print(str(err))
		sys.exit(2)

	if ('--filename' not in sys.argv[1:]):
		print("Error: You have to input file name. Use '--filename [string]' flag.")
		sys.exit(1)

	for opt, arg in opts:
		if (opt == '--filename'):
			filename = arg
		elif  (opt == '--outname'):
			outname = arg

	print("Reading file: " + filename)

	infile = open(filename,'rb')
	new_dict = pickle.load(infile)
	infile.close()
	samples, weights = new_dict.samples, np.exp(new_dict.logwt - new_dict.logz[-1])
	new_samples = dyfunc.resample_equal(samples, weights)

	for row in new_samples:
		u0 = row[0]
		tE = row[1]
		fbl = row[2]
		piEE = row[3]
		piEN = row[4]
		t0 = row[5] - 50000.0 + 0.5
		mag0 = row[6]

		row[0] = t0
		row[1] = tE
		row[2] = u0
		row[3] = piEN
		row[4] = piEE
		row[5] = mag0
		row[6] = fbl

	print(f"Saving DLC compatible data to: {outname}")
	np.save(outname, new_samples)
	print("Conversion complete!")

if __name__ == "__main__":
	main()