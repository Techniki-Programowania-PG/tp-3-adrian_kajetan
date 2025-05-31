import math
import scikit_build_example as sbe

def test_generate_sine():
    signal = sbe.generate_sine(10, 5.0, 100.0)
    assert isinstance(signal, list)
    assert len(signal) == 10
    assert all(isinstance(x, float) for x in signal)

def test_generate_cosine():
    signal = sbe.generate_cosine(10, 8.0, 100.0)
    assert len(signal) == 10

def test_generate_square():
    signal = sbe.generate_square(10, 6.0, 100.0)
    assert len(signal) == 10

def test_generate_sawtooth():
    signal = sbe.generate_sawtooth(10, 10.0, 100.0)
    assert len(signal) == 10

def test_dft_and_idft_identity():
    signal = sbe.generate_sine(10, 5.0, 100.0)
    spectrum = sbe.dft(signal)
    restored = sbe.idft(spectrum)
    for x, y in zip(signal, restored):
        assert math.isclose(x, y.real, rel_tol=1e-6)

def test_dft_output_type():
    signal = [1.0, 0.0, -1.0, 0.0]
    spectrum = sbe.dft(signal)
    assert all(isinstance(z, complex) for z in spectrum)

def test_idft_output_real():
    spectrum = [complex(1, 0), complex(0, 0), complex(-1, 0), complex(0, 0)]
    signal = sbe.idft(spectrum)
    assert all(isinstance(x, float) for x in signal)

def test_low_pass_filter_behavior():
    signal = [0.0, 1.0, 0.0, -1.0, 0.0]
    filtered = sbe.low_pass_filter(signal, 5.0, 100.0)
    assert len(filtered) == len(signal)

def test_detect_peaks_behavior():
    signal = [0.0, 1.0, 0.5, 2.0, 1.5, 0.0]
    peaks = sbe.detect_peaks(signal, 1.0)
    assert peaks == [1, 3]

def test_compute_energy_output():
    signal = [1.0, 2.0, 3.0]
    energy = sbe.compute_energy(signal)
    assert isinstance(energy, float)
    assert math.isclose(energy, 14.0, rel_tol=1e-6)

def test_plot_signal_executes():
    sbe.plot_signal([0.0, 1.0, 0.0, -1.0], title="Test Plot")

def test_moving_average_filter_2d():
    image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    filtered = sbe.moving_average_filter(image)
    assert len(filtered) == 3
    assert len(filtered[0]) == 3
