# DWD subtraction workflow

`eric_subtraction_all.py` is the entry point for subtraction across frequency
blocks in global rounds. Run it with `python eric_subtraction_all.py` in the
server scientific environment. Its three inputs are the observation's companion
orbit, the completed `eric_snr` result, and the simulated observation. The default
output is `data/dwd_subtraction_all.h5`; the default analysis covers all positive
Fourier frequencies through Nyquist with 10 microhertz blocks. Each block has a
fixed, zero-based index. Set `subtraction_frequency_range = (f_min, f_max)` in Hz
at the end of the script, or pass it to `run_subtraction`, to select the blocks
overlapping that interval. `None` selects the whole grid. The separate
`frequency_range` argument defines the grid itself; changing the subtraction
range does not renumber its blocks.

```python
result = run_subtraction(subtraction_frequency_range=(0.003, 0.004))
print(result['selected_block_indices'])
```

The first and last selected blocks use search limits clipped to the requested
interval. Source frequencies at the lower limit are included; the upper limit
is excluded except at the final grid edge. The HDF5 configuration records all
block indices and edges, the requested subtraction range, and the selected
indices. Blocks outside the interval are marked `outside_subtraction_range`;
they do not run F-statistic searches, RJMCMC or contribute fitted sources.

Each round holds one full-band residual fixed while the selected blocks are visited.
Covariance estimation, screening and one Eryn fit remain local to each block.
After the selected pass, accepted physical parameter rows from those blocks are
simulated together in the time domain, converted into native detector
measurements, combined by `tdi_gpu`, and subtracted from the original observation.
The next round uses a fresh FFT of that shared residual. Every block is screened
from the measured residual, even if the SNR catalogue contains no source there.
The shared FFT and measured spectral/noise context still cover the full
observation, including neighboring sidebands; the chosen indices determine
which blocks can enter fitting and supply sources for subtraction.
A multiscale spectral-excess screen determines which blocks enter a blind
four-amplitude F-statistic search over generic frequency, signed chirp and sky
trials. Blocks below that measured threshold avoid template generation and
sampling. They are checked again in every new round. The spectral screen is a
statistical gate with nominal, data-estimated noise probabilities; it does not
prove that a rejected block contains no source.

The SNR catalogue supplies post-fit labels only. Its absolute SNR, source counts,
parameters and IDs cannot decide whether to search or subtract a block. An
accepted source absent from that file receives a unique negative ID. Acceptance
uses fitted SNR, conditional F, carrier-band ownership and improvement of the
measured fit. Waveform overlap and acceptance conditional on already proposed
sources prevent neighboring blocks from subtracting the same sidebands twice.
Fitting itself still sees the same immutable residual in every block of a round.

The new pipeline samples a signed chirp coordinate, `df * T_obs**2`, so both
positive and negative derivatives are supported. The declared default search
range is -3e-12 to 3e-12 Hz/s, independent of catalogue values; override
`frequency_derivative_bounds` to change it. Bounds are recorded with the blocks.
This is a uniform prior in the signed frequency-derivative coordinate. F >= 16 and fitted SNR > 7 remain the default final screens.
Catalogue truth and the simulation-only ghost rule do not control acceptance in
the global controller. 

Keyword arguments to `run_subtraction` support smaller experiments without
editing production defaults. An optional `maximum_rounds` is recorded as a
resource limit. Outputs contain block indices and edges, selected indices,
per-round screening and sampler
checkpoints, and the final full-band signal and residual in time and Fourier
domains. `output/reports/subtraction_blind_20261012.md` records the revised
validation scope. The finite spectral/grid search is not an exhaustive search
of all possible DWD parameters. The earlier 0.324-second empty-catalogue result
in `output/reports/subtraction_all_20261012.md` belongs to the superseded
catalogue-gated implementation and does not measure this residual-only search.

## Review and current source

The detailed TeX/PDF review in `output/pdf/DWD injection and subtraction workflow` follows `eric_subtraction_all.py`, including indexed interval selection, blind measured-data search, one shared residual per global round, post-fit labels, and combined native time-domain subtraction. The observation producer and GPU/sampler helpers are documented as dependencies. The primary listing and its line references match the current 471-line controller.
