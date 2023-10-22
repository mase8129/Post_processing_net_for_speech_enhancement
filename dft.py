"""
Copyright 2023 ai|coustics. All Rights Reserved.
"""

import numpy as np
import tensorflow as tf


def _rdft_matrix(dft_length, halfsided=False):
    """Return a precalculated DFT matrix for positive frequencies only
    Args:
        dft_length (int) - DFT length
    Returns
        rdft_mat (k x n tensor) - precalculated DFT matrix rows are frequencies
            columns are samples, k dimension is dft_length // 2 +1 bins long
    """
    # freq bins
    if halfsided:
        k = np.arange(0, dft_length // 2 + 1)
    else:
        k = np.arange(0, dft_length)
    # Samples
    n = np.arange(0, dft_length)
    # complex frequency vector (now normalised to 2 pi)
    omega = -1j * 2.0 * np.pi / dft_length * k
    # complex phase, compute a matrix of value for the complex phase for each sample
    # location (n) and each freq bin (k) outer product If the two vectors have dimensions
    # k and n, then their outer product is an k × n matrix
    phase = np.outer(omega, n)
    # return transposed ready for matrix multiplication
    return np.exp(phase).astype(np.complex64).T


def get_window_fn(window_name=None):
    """Return a window function given its name.
    This function is used inside layers such as `STFT` to get a window function.
    Args:
        window_name (None or str): name of window function. On Tensorflow 2.3, there are five windows available in
        `tf.signal` (`hamming_window`, `hann_window`, `kaiser_bessel_derived_window`, `kaiser_window`, `vorbis_window`).
    """

    if window_name is None:
        return tf.signal.hann_window

    available_windows = {
        "hamming_window": tf.signal.hamming_window,
        "hann_window": tf.signal.hann_window,
    }
    if hasattr(tf.signal, "kaiser_bessel_derived_window"):
        available_windows[
            "kaiser_bessel_derived_window"
        ] = tf.signal.kaiser_bessel_derived_window
    if hasattr(tf.signal, "kaiser_window"):
        available_windows["kaiser_window"] = tf.signal.kaiser_window
    if hasattr(tf.signal, "vorbis_window"):
        available_windows["vorbis_window"] = tf.signal.vorbis_window

    if window_name not in available_windows:
        raise NotImplementedError(
            "Window name %s is not supported now. Currently, %d windows are"
            "supported - %s"
            % (
                window_name,
                len(available_windows),
                ", ".join([k for k in available_windows.keys()]),
            )
        )

    return available_windows[window_name]


class DFT(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fft=None,
        win_length=None,
        hop_length=None,
        window_name="hann_window",
        pad_begin=False,
        pad_end=False,
        input_data_format="channels_last",
        output_data_format="channels_last",
        **kwargs,
    ) -> None:
        """Computes STFT for a real-valued signal using DFT matrix.
        Real and imaginary parts are stored in extra dim,
        i.e., no complex dytypes are used.

        Args:
            n_fft (int): Number of FFTs. Defaults to `2048`
            win_length (int or None): Window length in sample. Defaults to `n_fft`.
            hop_length (int or None): Hop length in sample between analysis windows.
                Defaults to `n_fft // 4` following Librosa.
            window_name (str or None): *Name* of `tf.signal` function that returns a
                1D tensor window that is used in analysis.
                Defaults to `hann_window` which uses `tf.signal.hann_window`.
                Window availability depends on Tensorflow version.
                More details are at `kapre.backend.get_window()`.
            pad_begin (bool): Whether to pad with zeros along time axis
                (length: win_length - hop_length). Defaults to `False`.
            pad_end (bool): Whether to pad with zeros at the end of the signal.
            input_data_format (str): the audio data format of input waveform batch.
                `'channels_last'` if it's `(batch, time, channels)` and
                `'channels_first'` if it's `(batch, channels, time)`.
                Defaults to "channels last".
            output_data_format (str): The data format of output STFT.
                `'channels_last'` if you want `(batch, time, frequency, channels)` and
                `'channels_first'` if you want `(batch, channels, time, frequency)`.
                Defaults to "channels last".
            **kwargs: Keyword args for the parent keras layer (e.g., `name`)
        """
        super().__init__(**kwargs)
        if win_length is None:
            win_length = n_fft
        if hop_length is None:
            hop_length = win_length // 4

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_name = window_name
        self.window_fn = get_window_fn(window_name)
        self.pad_begin = pad_begin
        self.pad_end = pad_end
        self.input_data_format = input_data_format
        self.output_data_format = output_data_format

        dft_matrix_complex = _rdft_matrix(self.n_fft, halfsided=True)
        self.dft_matrix_real = tf.constant(
            np.real(dft_matrix_complex),
            dtype=self.compute_dtype,
            name="dft_matrix_real",
        )
        self.dft_matrix_imag = tf.constant(
            np.imag(dft_matrix_complex),
            dtype=self.compute_dtype,
            name="dft_matrix_imag",
        )

    def _prepare_signal(self, waveforms):
        """Frames and pads the input signal and multiplies with the window function.

        Args:
            waveforms (tf.Tensor): The input signal,
                shape: (B, N_SAMPLES, C) or (B, C, N_SAMPLES).

        Returns:
            framed_signal (tf.tensor): The framed and padded output signal,
                shape: (B, C, n_frames, win_length).
        """
        if self.input_data_format == "channels_last":
            waveforms = tf.transpose(
                waveforms, perm=(0, 2, 1)
            )  # always (batch, ch, time) from here

        if self.pad_begin:
            waveforms = tf.pad(
                waveforms,
                ([0, 0], [0, 0], [int(self.win_length - self.hop_length), 0]),
            )

        window = tf.reshape(
            self.window_fn(self.win_length, dtype=waveforms.dtype), [1, self.win_length]
        )
        framed_signal = tf.signal.frame(
            signal=waveforms,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            pad_end=self.pad_end,
            axis=-1,
        )
        framed_signal *= window
        return framed_signal

    def call(self, x):
        """
        Compute STFT of the input signal.
        Args:
            x (tf.Tensor): batch of audio signals, (
                batch, ch, time) or (batch, time, ch) based on input_data_format
        Return:
            spec (tf.Tensor): A STFT representation of x in a 2D batch shape. The last dimension is size two and contains
            the real and imaginary parts of the stft.
            Its shape is (B, T, F, C, 2) or (B. C, T, F, 2)
                depending on `output_data_format`.
                `T` is the number of frames, which is
                `((len_src + (win_length - hop_length) / hop_length) // win_length )`.
                if `pad_end` is `True`. `F` is the number of fft unique bins,
                which is `n_fft // 2 + 1` (the unique components of the FFT).
        """
        # (batch, ch, time) if input_data_format == 'channels_first'
        # (batch, time, ch) if input_data_format == 'channels_last'
        framed_signal = self._prepare_signal(x)
        spec_real = tf.matmul(framed_signal, self.dft_matrix_real)
        spec_imag = tf.matmul(framed_signal, self.dft_matrix_imag)
        spec = tf.stack([spec_real, spec_imag], axis=-1)

        if self.output_data_format == "channels_last":
            spec = tf.transpose(spec, perm=(0, 2, 3, 1, 4))  # (batch, t, f, ch, re/im)
        return spec

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_fft": self.n_fft,
                "win_length": self.win_length,
                "hop_length": self.hop_length,
                "window_name": self.window_name,
                "pad_begin": self.pad_begin,
                "pad_end": self.pad_end,
                "input_data_format": self.input_data_format,
                "output_data_format": self.output_data_format,
            }
        )
        return config
