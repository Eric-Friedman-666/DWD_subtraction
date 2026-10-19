"""Plot saved DWD Doppler posteriors; never simulate data or run inference."""
from pathlib import Path
import hashlib
import json

import corner
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtri
from scipy.stats import rankdata


PARAMETERS = ('log10_f', 'log10_df', 'longitude', 'latitude')
LABELS = (r'$\log_{10}(f/\mathrm{Hz})$',
          r'$\log_{10}(\dot f/[\mathrm{Hz}\,\mathrm{s}^{-1}])$',
          r'$\lambda\ [\mathrm{rad}]$', r'$\beta\ [\mathrm{rad}]$')


def bulk_ess(values):
    """Within-run rank-normalized initial-positive/monotone-sequence estimate."""
    n = len(values)
    if n < 8 or np.var(values) == 0:
        return np.nan
    values = ndtri((rankdata(values)-3/8)/(n+1/4))
    transform = np.fft.rfft(values-values.mean(), n=2*n)
    covariance = np.fft.irfft(transform*transform.conj(), n=2*n)[:n]
    correlation = covariance/covariance[0]
    pairs = []
    for i in range(0, n-1, 2):
        pair = correlation[i]+correlation[i+1]
        if pair < 0:
            break
        pairs.append(min(pair, pairs[-1] if pairs else pair))
    return float(n/max(1., -1+2*sum(pairs)))


def load_best_doppler_samples(result_file):
    result_file = Path(result_file)
    with result_file.open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    with h5py.File(result_file, 'r') as saved:
        chain = saved['rjmcmc/chains/0']
        q = chain['sampled_q'][:]
        counts = chain['source_count'][:]
        point = saved['recovery/doppler'][:]
        center = float(saved['rjmcmc/config/center'][()])
        duration = float(saved['rjmcmc/config/duration'][()])
        burn = int(saved['rjmcmc/config/burn_in'][()])
        thin = int(saved['rjmcmc/config/thin'][()])
        iterations = int(saved['rjmcmc/config/iterations'][()])
        modal = int(saved['rjmcmc/modal_count'][()])
        comparison = saved['injection_id_validation']
        ids = comparison['dwd_id'][:]
        overlap = comparison['overlap'][:]
        delta = comparison['delta'][:]
        recovered_index = comparison['recovered_index'][:]
        truth = np.column_stack([comparison['injected_parameters/'+key][:]
                                 for key in ('log10_f', 'log10_df', 'longitude', 'latitude')])
    assert modal == len(point)
    mask = counts == modal
    samples = q[mask, :modal].copy()
    assert len(samples) > 10 and np.isfinite(samples).all()
    scale = center*duration*np.log(10.)
    samples[:, :, 0] = np.log10(center)+samples[:, :, 0]/scale
    assert np.max(np.abs(samples[:, :, 3])) <= 1.+1.e-12
    samples[:, :, 3] = np.arcsin(np.clip(samples[:, :, 3], -1., 1.))

    # Relabel against the recovered catalogue, not against injection truth.
    permutations = np.empty((len(samples), modal), dtype=int)
    for i, values in enumerate(samples):
        longitude_offset = (values[:, None, 2]-point[None, :, 2]+np.pi)%(2*np.pi)-np.pi
        cost = ((10.**values[:, None, 0]-point[None, :, 0])*duration)**2
        cost += (longitude_offset/.15)**2+((values[:, None, 3]-point[None, :, 3])/.15)**2
        rows, columns = linear_sum_assignment(cost)
        order = rows[np.argsort(columns)]
        permutations[i] = order
        samples[i] = values[order].copy()
    samples[:, :, 2] = point[None, :, 2]+(samples[:, :, 2]-point[None, :, 2]+np.pi)%(2*np.pi)-np.pi
    production_indices = np.flatnonzero(mask)
    cycles = burn+1+np.arange(len(counts))*thin
    assert cycles[-1] <= iterations

    # ESS uses one uninterrupted visit, never concatenates different-N gaps.
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    longest = np.argmax(stops-starts)
    diagnostic = (production_indices >= starts[longest]) & (production_indices < stops[longest])
    eligible = np.flatnonzero(np.isfinite(overlap) & np.isfinite(delta))
    ranking = eligible[np.lexsort((ids[eligible], delta[eligible], -overlap[eligible]))][:2]
    records = []
    for row in ranking:
        component = int(recovered_index[row])
        injected = truth[row].copy()
        injected[2] = point[component, 2]+(injected[2]-point[component, 2]+np.pi)%(2*np.pi)-np.pi
        best_fit = point[component].copy()
        best_fit[:2] = np.log10(best_fit[:2])
        draws = samples[:, component].copy()
        records.append(dict(dwd_id=int(ids[row]), recovered_index=component,
            overlap=float(overlap[row]), delta=float(delta[row]), samples=draws,
            injected=injected, best_fit=best_fit,
            quantiles=np.quantile(draws, [.16, .5, .84], axis=0),
            bulk_ess=np.array([bulk_ess(draws[diagnostic, j]) for j in range(4)])))
    metadata = dict(source_file=str(result_file.resolve()), source_sha256=source_hash,
        parameters=PARAMETERS, sample_count=len(samples), production_draws=len(counts),
        burn_in=burn, iterations=iterations, modal_count=modal,
        modal_probability=float(mask.mean()), production_cycles=cycles[mask],
        relabeled_draws=int(np.any(permutations != np.arange(modal), axis=1).sum()),
        ess_contiguous_draws=int(diagnostic.sum()),
        target='DWD recovery conditional on a fixed, data-estimated effective covariance.',
        selection='Two highest final injected-versus-recovered waveform overlaps O.',
        independent_chain_convergence_assessed=False)
    return records, metadata


def plot_best_doppler_corners(result_file, output_directory='./data/doppler_corners'):
    records, metadata = load_best_doppler_samples(result_file)
    output_directory = Path(output_directory)/metadata['source_sha256'][:12]
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = []
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.grid':False, 'figure.facecolor':'white', 'axes.facecolor':'white'}):
        for record in records:
            values, injected = record['samples'], record['injected']
            low = np.minimum(values.min(axis=0), injected)
            high = np.maximum(values.max(axis=0), injected)
            span = high-low
            ranges = list(zip(low-.04*span, high+.04*span))
            figure = plt.figure(figsize=(10.4, 10.2))
            corner.corner(values, fig=figure, bins=35, range=ranges,
                labels=LABELS, label_kwargs={'fontsize':11}, color='#2776B9',
                levels=(.68, .95), smooth=.7, plot_datapoints=False,
                plot_density=False, fill_contours=True, max_n_ticks=3,
                hist_kwargs={'density':True, 'linewidth':1.4})
            figure.subplots_adjust(left=.13, bottom=.18, right=.98, top=.86, wspace=.13, hspace=.13)
            axes = np.array(figure.axes).reshape(4, 4)
            formats = ('%.8f', '%.1f', '%.3f', '%.3f')
            for row in range(4):
                for col in range(row+1):
                    axis = axes[row, col]
                    axis.axvline(injected[col], color='#343A40', ls='--', lw=1.1)
                    axis.xaxis.set_major_locator(MaxNLocator(2, prune='both'))
                    axis.xaxis.set_major_formatter(FormatStrFormatter(formats[col]))
                    axis.tick_params(axis='x', labelsize=8 if col == 0 else 9,
                                     labelbottom=row == 3, labelrotation=30 if col == 0 else 0)
                    axis.tick_params(axis='y', labelsize=9, labelleft=row > col and col == 0,
                                     labelrotation=0)
                    if row == 3:
                        axis.xaxis.set_label_coords(.5, -.42)
                    if row > col:
                        axis.axhline(injected[row], color='#343A40', ls='--', lw=1.1)
                        axis.plot(injected[col], injected[row], '+', color='#343A40', ms=6, mew=1.1)
                        axis.yaxis.set_major_formatter(FormatStrFormatter(formats[row]))
            figure.suptitle(f"DWD {record['dwd_id']} | Four-Doppler-parameter posterior",
                x=.13, y=.975, ha='left', fontsize=16)
            figure.text(.13, .935,
                f"O = {record['overlap']:.6f}   |   delta = {record['delta']:.6f}   |   "
                f"{len(values):,} cold post-burn-in draws; N = {metadata['modal_count']}", fontsize=11)
            figure.legend(handles=[Line2D([], [], color='#2776B9', lw=1.5, label='Posterior: 68% / 95% contours'),
                                   Line2D([], [], color='#343A40', ls='--', lw=1.1, label='Injected values')],
                          loc='upper right', bbox_to_anchor=(.965, .86), frameon=False, fontsize=10)
            quantiles = record['quantiles']
            names = ('log10 f', 'log10 fdot', 'lambda', 'beta')
            statistics = ['Median [16th, 84th percentiles]']
            for j, name in enumerate(names):
                digits = 8 if j == 0 else (3 if j == 1 else 5)
                statistics += [f"{name}: {quantiles[1,j]:.{digits}f}",
                               f"  [{quantiles[0,j]:.{digits}f}, {quantiles[2,j]:.{digits}f}]"]
            figure.text(.61, .73, '\n'.join(statistics), va='top', fontsize=9.5, linespacing=1.45)
            figure.text(.13, .035,
                'Fixed data-estimated XYZ covariance; source-count-conditional posterior.\n'
                f"Single ladder; convergence not established. Minimum within-run bulk ESS: {np.nanmin(record['bulk_ess']):.1f}.",
                fontsize=9, color='#555A61')
            stem = output_directory/f"dwd_{record['dwd_id']}_doppler_corner"
            figure.savefig(stem.with_suffix('.png'), dpi=160, bbox_inches='tight')
            figure.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
            record['png'] = str(stem.with_suffix('.png'))
            record['pdf'] = str(stem.with_suffix('.pdf'))
            figures.append(figure)
    with h5py.File(output_directory/'corner_samples.h5', 'w') as saved:
        saved.attrs['source_file'] = metadata['source_file']
        saved.attrs['source_sha256'] = metadata['source_sha256']
        saved.attrs['target'] = metadata['target']
        saved.attrs['modal_count'] = metadata['modal_count']
        saved.attrs['modal_probability'] = metadata['modal_probability']
        saved.attrs['parameters'] = json.dumps(PARAMETERS)
        saved['production_cycles'] = metadata['production_cycles']
        for record in records:
            group = saved.create_group(str(record['dwd_id']))
            for name in ('samples', 'injected', 'best_fit', 'quantiles', 'bulk_ess'):
                group[name] = record[name]
            for name in ('overlap', 'delta', 'recovered_index'):
                group.attrs[name] = record[name]
    summary = {key:value for key,value in metadata.items() if key != 'production_cycles'}
    summary['sources'] = [{key:value.tolist() if isinstance(value, np.ndarray) else value
                           for key,value in record.items() if key != 'samples'} for record in records]
    (output_directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return figures, summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/10dwd_subtraction.h5')
    parser.add_argument('--output', default='./data/doppler_corners')
    args = parser.parse_args()
    figures, summary = plot_best_doppler_corners(args.input, args.output)
    print(json.dumps(summary, indent=2))
    for figure in figures:
        plt.close(figure)
