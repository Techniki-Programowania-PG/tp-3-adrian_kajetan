#define _USE_MATH_DEFINES

#include "fourier.hpp"
#include <vector>
#include <complex>
#include <cmath>

std::vector<std::complex<double>> dft(const std::vector<double>& inputSignal) {
	size_t size = inputSignal.size();
	std::vector<std::complex<double>> outputSpectrum(size);

	for (size_t k = 0; k < size; ++k) {
		std::complex<double> value(0.0, 0.0);
		for (size_t n = 0; n < size; ++n) {
			double theta = -2.0 * M_PI * k * n / size;
			value += inputSignal[n] * std::polar(1.0, theta);  // e^{-jθ}
		}
		outputSpectrum[k] = value;
	}

	return outputSpectrum;
}

std::vector<double> idft(const std::vector<std::complex<double>>& inputSpectrum) {
	size_t size = inputSpectrum.size();
	std::vector<double> outputSignal(size);

	for (size_t n = 0; n < size; ++n) {
		std::complex<double> value(0.0, 0.0);
		for (size_t k = 0; k < size; ++k) {
			double theta = 2.0 * M_PI * k * n / size;
			value += inputSpectrum[k] * std::polar(1.0, theta);  // e^{jθ}
		}
		outputSignal[n] = value.real() / static_cast<double>(size);
	}

	return outputSignal;
}
