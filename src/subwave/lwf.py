from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from .tensor import AxisAnnotatedTensor

_LWF_MAGIC = b"LWF1"
_LWF_VERSION = 3

# ---------------------------------------------------------------------------
# Binary primitives
# ---------------------------------------------------------------------------

def _read_u32(f) -> int:
    return struct.unpack('<I', f.read(4))[0]

def _read_i32(f) -> int:
    return struct.unpack('<i', f.read(4))[0]

def _read_u64(f) -> int:
    return struct.unpack('<Q', f.read(8))[0]

def _read_f64(f) -> float:
    return struct.unpack('<d', f.read(8))[0]

def _read_string(f) -> str:
    n = _read_u32(f)
    return f.read(n).decode('utf-8') if n else ''

def _decode_int16(raw: bytes, phys_min: float, phys_max: float) -> np.ndarray:
    digital = np.frombuffer(raw, dtype='<i2').astype(np.float64)
    return (digital + 32768.0) * (phys_max - phys_min) / 65535.0 + phys_min

# ---------------------------------------------------------------------------
# Header reader  (cursor lands at start of index section on return)
# ---------------------------------------------------------------------------

def _read_header(f) -> dict:
    magic = f.read(4)
    if magic != _LWF_MAGIC:
        raise ValueError(f"not a valid .lwf file (bad magic: {magic!r})")
    version = _read_i32(f)
    if version != _LWF_VERSION:
        raise ValueError(f"unsupported .lwf version {version} (expected {_LWF_VERSION})")

    id_   = _read_string(f)
    _edf  = _read_string(f)
    _read_string(f)          # stored output filename — discard
    startdate = _read_string(f)
    starttime = _read_string(f)
    tag   = _read_string(f)
    align = _read_string(f)

    n_annots = _read_i32(f)
    annots = [_read_string(f) for _ in range(n_annots)]

    n_ch = _read_i32(f)
    channels = []
    for _ in range(n_ch):
        label    = _read_string(f)
        unit     = _read_string(f)
        _read_u64(f)          # sample_step_tp — not needed in Python
        sr       = _read_f64(f)
        phys_min = _read_f64(f)
        phys_max = _read_f64(f)
        channels.append({'label': label, 'unit': unit, 'sr': sr,
                         'phys_min': phys_min, 'phys_max': phys_max})

    n_features = _read_i32(f)
    feature_names = [_read_string(f) for _ in range(n_features)]

    n_waves = _read_i32(f)

    return {
        'id':            id_,
        'startdate':     startdate,
        'starttime':     starttime,
        'tag':           tag,
        'align':         align,
        'annots':        annots,
        'channels':      channels,
        'feature_names': feature_names,
        'n_waves':       n_waves,
    }

# ---------------------------------------------------------------------------
# Events reader  (index section then payload section, single sequential pass)
# ---------------------------------------------------------------------------

def _read_events(f, header: dict) -> tuple[np.ndarray, pd.DataFrame]:
    n_waves    = header['n_waves']
    n_ch       = len(header['channels'])
    n_features = len(header['feature_names'])

    # build label -> (phys_min, phys_max) for int16 decoding
    ch_phys = {c['label']: (c['phys_min'], c['phys_max']) for c in header['channels']}

    # --- pass 1: index section ---
    index_rows: list[dict] = []
    expected_n: int | None = None
    annot_ch_match = False  # detected if any event has fewer blocks than channels

    for w in range(n_waves):
        annot    = _read_string(f)
        instance = _read_string(f)
        annot_ch = _read_string(f)
        annot_start = _read_f64(f)
        annot_stop  = _read_f64(f)
        anchor      = _read_f64(f)
        wave_start  = _read_f64(f)
        wave_stop   = _read_f64(f)
        _read_u64(f)              # payload_offset — not needed (sequential read)

        n_blocks = _read_i32(f)
        if n_blocks < n_ch:
            annot_ch_match = True

        ns_list: list[int] = []
        for _ in range(n_blocks):
            ns = _read_i32(f)
            _read_f64(f)          # data_start_sec
            _read_f64(f)          # data_stop_sec
            ns_list.append(ns)

        ns = ns_list[0] if ns_list else 0

        if expected_n is None:
            expected_n = ns
        elif ns != expected_n:
            raise ValueError(
                f"non-uniform waveform length at event {w}: "
                f"got {ns} samples, expected {expected_n}"
            )

        index_rows.append({
            'annot':           annot,
            'instance':        instance,
            'annot_ch':        annot_ch,
            'annot_start_sec': annot_start,
            'annot_stop_sec':  annot_stop,
            'anchor_sec':      anchor,
            'wave_start_sec':  wave_start,
            'wave_stop_sec':   wave_stop,
            'n_blocks':        n_blocks,
        })

    if expected_n is None:
        expected_n = 0

    # output channels: 1 per event (annot-ch-match) or all header channels
    out_n_ch = 1 if annot_ch_match else n_ch
    data = np.full((n_waves, out_n_ch, expected_n), np.nan, dtype=np.float64)
    meta_col: list[str] = []

    # --- pass 2: payload section ---
    for w in range(n_waves):
        meta_col.append(_read_string(f))  # meta — only in payload
        n_blocks = _read_i32(f)

        for b in range(n_blocks):
            if n_features > 0:
                f.read(4 + n_features * 8)   # feature_qc (int32) + feature values (float64)

            ns = index_rows[w]['n_blocks']   # same as n_blocks, use index value
            ns = expected_n                   # uniform (validated above)
            raw = f.read(ns * 2)             # int16 = 2 bytes

            # determine which channel this block belongs to
            if annot_ch_match:
                label = index_rows[w]['annot_ch']
            else:
                label = header['channels'][b]['label']

            phys_min, phys_max = ch_phys.get(label, (0.0, 1.0))
            data[w, b, :] = _decode_int16(raw, phys_min, phys_max)

    event_meta = pd.DataFrame(index_rows).drop(columns=['n_blocks'])
    event_meta['meta'] = meta_col
    return data, event_meta, annot_ch_match

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_paths(
    paths: Union[str, Path, Iterable],
    recur: bool,
) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    resolved: list[Path] = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            pattern = '**/*.lwf' if recur else '*.lwf'
            resolved.extend(sorted(p.glob(pattern)))
        else:
            resolved.append(p)

    if not resolved:
        raise ValueError("no .lwf files found in the given paths")
    return resolved

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lwf_summary(
    paths: Union[str, Path, Iterable],
    *,
    recur: bool = False,
) -> pd.DataFrame:
    """Summarise one or more .lwf files without loading signal data.

    Parameters
    ----------
    paths:
        A .lwf file path, a directory containing .lwf files, or a list of
        either.
    recur:
        Recurse into subdirectories when *paths* contains a directory.

    Returns
    -------
    pandas.DataFrame with one row per file and columns:
    ``file``, ``id``, ``tag``, ``startdate``, ``starttime``, ``align``,
    ``n_waves``, ``n_channels``, ``channels``, ``srs``, ``annots``,
    ``n_features``.
    """
    files = _resolve_paths(paths, recur)
    rows = []
    for p in files:
        with open(p, 'rb') as f:
            h = _read_header(f)
        rows.append({
            'file':       str(p),
            'id':         h['id'],
            'tag':        h['tag'],
            'startdate':  h['startdate'],
            'starttime':  h['starttime'],
            'align':      h['align'],
            'n_waves':    h['n_waves'],
            'n_channels': len(h['channels']),
            'channels':   ','.join(c['label'] for c in h['channels']),
            'srs':        ','.join(str(c['sr']) for c in h['channels']),
            'annots':     ','.join(h['annots']),
            'n_features': len(h['feature_names']),
        })
    return pd.DataFrame(rows)


def from_lwf(
    paths: Union[str, Path, Iterable],
    *,
    recur: bool = False,
) -> AxisAnnotatedTensor:
    """Load one or more .lwf files into an AxisAnnotatedTensor.

    All files must share the same channel labels and per-channel sample rates.
    All waveform events must have the same number of samples (i.e. Luna's
    ``require=full`` mode, or any mode that produced uniform windows).

    When the file was written with ``annot-ch-match=T``, each event contains
    one channel block (the detecting channel).  The tensor shape is then
    ``(n_waves, 1, n_samples)`` and ``axis_meta['instance']['annot_ch']``
    identifies the channel for each event.

    Parameters
    ----------
    paths:
        A .lwf file path, a directory containing .lwf files, or a list of
        either.
    recur:
        Recurse into subdirectories when *paths* contains a directory.

    Returns
    -------
    AxisAnnotatedTensor with axes ``['instance', 'channel', 'sample']``.
    """
    files = _resolve_paths(paths, recur)

    # --- pass 1: read all headers and validate consistency ---
    headers: list[dict] = []
    for p in files:
        with open(p, 'rb') as f:
            h = _read_header(f)
        h['_path'] = p
        headers.append(h)

    ref_ch = headers[0]['channels']

    ref_sig = [(c['label'], c['sr']) for c in ref_ch]
    for h in headers[1:]:
        sig = [(c['label'], c['sr']) for c in h['channels']]
        if sig != ref_sig:
            raise ValueError(
                f"channel mismatch:\n"
                f"  {headers[0]['_path']}: {ref_sig}\n"
                f"  {h['_path']}: {sig}"
            )

    all_srs = [c['sr'] for c in ref_ch]
    if len(set(all_srs)) > 1:
        raise ValueError(
            f"channels have mixed sample rates {all_srs}; "
            f"select a single channel before loading"
        )
    sr = all_srs[0]

    ref_align = headers[0]['align']

    # --- pass 2: read events from each file ---
    all_data:  list[np.ndarray] = []
    all_meta:  list[pd.DataFrame] = []
    expected_n: int | None = None
    detected_annot_ch_match: bool | None = None

    for h in headers:
        with open(h['_path'], 'rb') as f:
            _read_header(f)
            data, event_meta, annot_ch_match = _read_events(f, h)

        if detected_annot_ch_match is None:
            detected_annot_ch_match = annot_ch_match
        elif annot_ch_match != detected_annot_ch_match:
            raise ValueError(
                f"mixed annot-ch-match modes across files: {h['_path']}"
            )

        ns = data.shape[2]
        if expected_n is None:
            expected_n = ns
        elif ns != expected_n:
            raise ValueError(
                f"sample count mismatch across files: "
                f"expected {expected_n}, got {ns} in {h['_path']}"
            )

        event_meta.insert(0, 'file', str(h['_path']))
        event_meta.insert(0, 'tag',  h['tag'])
        event_meta.insert(0, 'id',   h['id'])

        all_data.append(data)
        all_meta.append(event_meta)

    if expected_n is None:
        expected_n = 0
    if detected_annot_ch_match is None:
        detected_annot_ch_match = False

    data       = np.concatenate(all_data, axis=0)
    event_meta = pd.concat(all_meta, ignore_index=True)

    n_events, n_ch_out, n_samples = data.shape

    if n_events > 0:
        offset = (
            event_meta['wave_start_sec'].iloc[0]
            - event_meta['anchor_sec'].iloc[0]
        )
    else:
        offset = 0.0
    time_axis = offset + np.arange(n_samples) / sr

    # channel axis metadata
    if detected_annot_ch_match:
        ch_meta = pd.DataFrame([{'label': '(annot_ch)', 'unit': '', 'sr': sr}])
    else:
        ch_meta = pd.DataFrame({
            'label': [c['label'] for c in ref_ch],
            'unit':  [c['unit']  for c in ref_ch],
            'sr':    [c['sr']    for c in ref_ch],
        })

    col_order = [
        'id', 'tag', 'file',
        'annot', 'instance', 'annot_ch', 'meta',
        'anchor_sec', 'annot_start_sec', 'annot_stop_sec',
        'wave_start_sec', 'wave_stop_sec',
    ]
    event_meta = event_meta[col_order].reset_index(drop=True)

    return AxisAnnotatedTensor(
        data=data,
        axes=['instance', 'channel', 'sample'],
        axis_index={
            'instance': np.arange(n_events),
            'channel':  np.arange(n_ch_out),
            'sample':   time_axis,
        },
        axis_meta={
            'instance': event_meta,
            'channel':  ch_meta,
            'sample': pd.DataFrame({
                'sample_index': np.arange(n_samples),
                'time':         time_axis,
            }),
        },
        attrs={
            'sfreq':             sr,
            'source_files':      [str(h['_path']) for h in headers],
            'align':             ref_align,
            'annot_ch_match':    detected_annot_ch_match,
        },
    )
