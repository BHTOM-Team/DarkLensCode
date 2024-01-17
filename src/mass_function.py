import numpy as np


class MassFunction:
	def __init__(self,
				 mass_pows: np.array,
				 mass_breaks: np.array
				 ) -> None:
		"""
		Initializes MassFunction object.

		Agrs:
		mass_pows: Array of powers of the mass function.
		mass_breaks: Array of breaks in the mass function.
		"""
		self.mass_pows = mass_pows
		self.mass_breaks = mass_breaks

		if len(mass_pows) != len(mass_breaks):
			raise ValueError("List of mass powers has to be the same lenght as the list of breaks.")

		self.norms = [1]
		for i in range(1, len(mass_pows)):
			norm = self.norms[-1] * (mass_breaks[i - 1] ** mass_pows[i - 1]) / (mass_breaks[i - 1] ** mass_pows[i])
			self.norms.append(norm)
		self.norms = np.array(self.norms)


	def calc_prob(self,
				  mass:float
				  ) -> float:
		"""
		Calculates unnormalised probability of a given mass.

		Args:
			mass: Mass for calculating a probability

		Returns:
			Probability of getting an object of certain mass according to mass-function
			described by mass_pows and mass_break.
		"""
		idx = np.where(mass < self.mass_breaks)
		return (mass**self.mass_pows[idx][0]) * self.norms[idx][0]
