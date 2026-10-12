"""Load response-ready LISA DWD parameters from the public LDC catalogue."""

from functools import lru_cache
from contextlib import contextmanager
import json
import os
from pathlib import Path

import h5py
import numpy as np
from astropy import units as u
from astropy.coordinates import (
    BarycentricMeanEcliptic,
    CartesianRepresentation,
    SkyCoord,
)
from astropy.time import Time
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits


@lru_cache(maxsize=1)
def _bme_to_icrs_rotation() -> np.ndarray:
    """Return the fixed BME-J2000 to ICRS Cartesian rotation matrix."""
    frame = BarycentricMeanEcliptic(equinox=Time("J2000"))
    columns = []
    for axis in np.eye(3):
        coordinate = SkyCoord(
            CartesianRepresentation(*(axis * u.one)),
            frame=frame,
        ).icrs
        columns.append(coordinate.cartesian.xyz.value)
    return np.column_stack(columns)


def _bme_to_icrs(
    longitude: np.ndarray,
    latitude: np.ndarray,
    polarization: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert LDC BME positions and polarization angles to ICRS."""
    rotation = _bme_to_icrs_rotation()
    cos_longitude = np.cos(longitude)
    sin_longitude = np.sin(longitude)
    cos_latitude = np.cos(latitude)
    sin_latitude = np.sin(latitude)
    direction_bme = np.column_stack(
        (
            cos_latitude * cos_longitude,
            cos_latitude * sin_longitude,
            sin_latitude,
        )
    )
    u_bme = np.column_stack(
        (sin_longitude, -cos_longitude, np.zeros_like(longitude))
    )
    direction_icrs = direction_bme @ rotation.T
    transported_u = u_bme @ rotation.T

    ra = np.mod(
        np.arctan2(direction_icrs[:, 1], direction_icrs[:, 0]),
        2.0 * np.pi,
    )
    dec = np.arcsin(np.clip(direction_icrs[:, 2], -1.0, 1.0))
    u_icrs = np.column_stack(
        (np.sin(ra), -np.cos(ra), np.zeros_like(ra))
    )
    v_icrs = np.column_stack(
        (
            -np.sin(dec) * np.cos(ra),
            -np.sin(dec) * np.sin(ra),
            np.cos(dec),
        )
    )
    basis_rotation = np.arctan2(
        np.einsum("ij,ij->i", transported_u, v_icrs),
        np.einsum("ij,ij->i", transported_u, u_icrs),
    )
    psi = np.mod(polarization + basis_rotation, np.pi)

    return ra, dec, psi


def _rows_to_catalog(
    selected_rows: np.ndarray,
    ldc_row_index: np.ndarray,
) -> dict[str, np.ndarray]:
    ra, dec, psi = _bme_to_icrs(
        selected_rows["EclipticLongitude"],
        selected_rows["EclipticLatitude"],
        selected_rows["Polarization"],
    )

    return {
        "source_id": selected_rows["Name"],
        "ldc_row_index": ldc_row_index,
        "A": selected_rows["Amplitude"],
        "f": selected_rows["Frequency"],
        "df": selected_rows["FrequencyDerivative"],
        "phi0": selected_rows["InitialPhase"],
        "iota": selected_rows["Inclination"],
        "psi": psi,
        "ra": ra,
        "dec": dec,
    }


def strongest_dwd_catalog(
    number_sources: int,
    ldc_file: str | Path,
    catalog_path: str = "/sky/dgb/cat",
    chunk_rows: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Read the strongest finite-amplitude DWDs directly from an LDC file."""
    number_sources = int(number_sources)
    top_amplitudes = np.empty(0, dtype=np.float64)
    top_row_indices = np.empty(0, dtype=np.int64)

    with h5py.File(ldc_file, "r") as hdf5:
        ldc_catalog = hdf5[catalog_path]
        catalog_rows = len(ldc_catalog)
        progress = tqdm(
            total=catalog_rows,
            desc="Selecting strongest DWDs",
            unit="row",
            unit_scale=True,
            dynamic_ncols=True,
        )
        for start in range(0, catalog_rows, chunk_rows):
            stop = min(start + chunk_rows, catalog_rows)
            amplitudes = np.asarray(
                ldc_catalog.fields("Amplitude")[start:stop]
            ).reshape(-1)
            valid_indices = np.flatnonzero(np.isfinite(amplitudes))
            chunk_amplitudes = amplitudes[valid_indices]
            chunk_row_indices = start + valid_indices

            if len(chunk_amplitudes) > number_sources:
                local_indices = np.argpartition(
                    chunk_amplitudes,
                    -number_sources,
                )[-number_sources:]
                chunk_amplitudes = chunk_amplitudes[local_indices]
                chunk_row_indices = chunk_row_indices[local_indices]

            candidate_amplitudes = np.concatenate(
                (top_amplitudes, chunk_amplitudes)
            )
            candidate_row_indices = np.concatenate(
                (top_row_indices, chunk_row_indices)
            )
            if len(candidate_amplitudes) > number_sources:
                strongest = np.argpartition(
                    candidate_amplitudes,
                    -number_sources,
                )[-number_sources:]
                top_amplitudes = candidate_amplitudes[strongest]
                top_row_indices = candidate_row_indices[strongest]
            else:
                top_amplitudes = candidate_amplitudes
                top_row_indices = candidate_row_indices
            progress.update(stop - start)
        progress.close()

        amplitude_order = np.argsort(top_amplitudes)[::-1]
        ldc_row_index = top_row_indices[amplitude_order]
        read_order = np.argsort(ldc_row_index)
        rows_in_file_order = ldc_catalog[ldc_row_index[read_order], 0]

    selected_rows = np.empty_like(rows_in_file_order)
    selected_rows[read_order] = rows_in_file_order
    return _rows_to_catalog(selected_rows, ldc_row_index)


def dwd_ids_to_catalog(
    dwd_ids: np.ndarray,
    ldc_file: str | Path,
    catalog_path: str = "/sky/dgb/cat",
    chunk_rows: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Resolve LDC DWD IDs into response-ready source parameters.

    The returned arrays retain the input ID order. Sky position and
    polarization are converted from the LDC BME-J2000 basis to ICRS for
    ``lisagwresponse.GalacticBinary``.
    """
    dwd_ids = np.asarray(dwd_ids, dtype=np.int64).reshape(-1)
    id_position = {
        int(dwd_id): index
        for index, dwd_id in enumerate(dwd_ids)
    }
    ldc_row_index = np.full(len(dwd_ids), -1, dtype=np.int64)

    with h5py.File(ldc_file, "r") as hdf5:
        ldc_catalog = hdf5[catalog_path]
        progress = tqdm(
            range(0, len(ldc_catalog), chunk_rows),
            desc="Locating DWD IDs",
        )
        for start in progress:
            stop = min(start + chunk_rows, len(ldc_catalog))
            chunk_ids = np.asarray(
                ldc_catalog.fields("Name")[start:stop]
            ).reshape(-1)
            matched = np.flatnonzero(np.isin(chunk_ids, dwd_ids))
            for local_index in matched:
                output_index = id_position[int(chunk_ids[local_index])]
                ldc_row_index[output_index] = start + local_index
            if np.all(ldc_row_index >= 0):
                progress.update(progress.total - progress.n)
                break

        read_order = np.argsort(ldc_row_index)
        rows_in_file_order = ldc_catalog[ldc_row_index[read_order], 0]

    selected_rows = np.empty_like(rows_in_file_order)
    selected_rows[read_order] = rows_in_file_order
    return _rows_to_catalog(selected_rows, ldc_row_index)


def _save_tree(group, values):
    for name, value in values.items():
        if isinstance(value, dict):
            _save_tree(group.create_group(name), value)
        elif value is not None:
            array = np.asarray(value)
            if array.dtype.kind in 'UO':
                group.attrs[name] = json.dumps(value)
            else:
                group.create_dataset(name, data=array,
                    **({'compression': 'gzip', 'shuffle': True} if array.ndim and array.size > 256 else {}))


@contextmanager
def single_cpu():
    """Respect a block's taskset assignment; never launch a process pool."""
    affinity = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
    if affinity is not None:
        os.sched_setaffinity(0, {min(affinity)})
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)
