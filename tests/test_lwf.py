"""Tests for the new LWF v3 format (int16 samples, phys_min/phys_max in header,
payload contains only meta + n_blocks + samples)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subwave.lwf import _read_header, from_lwf, lwf_summary

# ---------------------------------------------------------------------------
# Synthetic LWF writer (mirrors the C++ write_* functions)
# ---------------------------------------------------------------------------

_MAGIC   = b"LWF1"
_VERSION = 3


def _wu32(v: int) -> bytes:
    return struct.pack('<I', v)

def _wi32(v: int) -> bytes:
    return struct.pack('<i', v)

def _wu64(v: int) -> bytes:
    return struct.pack('<Q', v)

def _wf64(v: float) -> bytes:
    return struct.pack('<d', v)

def _wstr(s: str) -> bytes:
    b = s.encode('utf-8')
    return _wu32(len(b)) + b

def _encode_int16(values: np.ndarray, phys_min: float, phys_max: float) -> bytes:
    gain   = 65535.0 / (phys_max - phys_min) if phys_max != phys_min else 1.0
    offset = -32768.0 - gain * phys_min
    dv = np.clip(np.round(gain * values + offset), -32768, 32767).astype(np.int16)
    return dv.tobytes()


def make_lwf(
    path: Path,
    *,
    id_: str = "TEST01",
    tag: str = "test",
    align: str = "mid",
    annots: list[str] | None = None,
    channels: list[dict] | None = None,
    waves: list[dict] | None = None,
    annot_ch_match: bool = False,
) -> None:
    """Write a minimal synthetic .lwf file in the current format.

    Each entry in *waves* is a dict with keys:
      annot, instance, annot_ch, annot_start_sec, annot_stop_sec,
      anchor_sec, wave_start_sec, wave_stop_sec, meta,
      blocks: list of {'label': str, 'values': np.ndarray}
    """
    annots   = annots   or ["SO_neg_pk"]
    channels = channels or [{'label': 'CZ', 'unit': 'uV', 'sr': 128.0,
                              'phys_min': -200.0, 'phys_max': 200.0}]
    waves    = waves    or []

    n_ch = len(channels)

    # ---- header bytes ----
    hdr = (
        _MAGIC
        + _wi32(_VERSION)
        + _wstr(id_)
        + _wstr("test.edf")
        + _wstr(str(path))
        + _wstr("01.01.24")
        + _wstr("00.00.00")
        + _wstr(tag)
        + _wstr(align)
    )
    hdr += _wi32(len(annots))
    for a in annots:
        hdr += _wstr(a)

    hdr += _wi32(n_ch)
    sample_step_tp = int(round(1e9 / channels[0]['sr']))
    for ch in channels:
        hdr += (_wstr(ch['label']) + _wstr(ch['unit'])
                + _wu64(sample_step_tp) + _wf64(ch['sr'])
                + _wf64(ch['phys_min']) + _wf64(ch['phys_max']))

    hdr += _wi32(0)            # n_features = 0
    hdr += _wi32(len(waves))

    # ---- index bytes ----
    ch_map = {c['label']: c for c in channels}

    index_bytes = b""
    for w in waves:
        blocks = w['blocks']
        n_blocks = len(blocks)
        index_bytes += (
            _wstr(w['annot'])
            + _wstr(w.get('instance', '.'))
            + _wstr(w.get('annot_ch', '.'))
            + _wf64(w.get('annot_start_sec', 0.0))
            + _wf64(w.get('annot_stop_sec',  0.0))
            + _wf64(w.get('anchor_sec',       0.0))
            + _wf64(w.get('wave_start_sec',  -1.5))
            + _wf64(w.get('wave_stop_sec',    1.5))
            + _wu64(0)         # payload_offset placeholder
            + _wi32(n_blocks)
        )
        for blk in blocks:
            ns = len(blk['values'])
            index_bytes += _wi32(ns) + _wf64(-1.5) + _wf64(1.5)

    # ---- payload bytes ----
    payload_bytes = b""
    for w in waves:
        blocks = w['blocks']
        payload_bytes += _wstr(w.get('meta', ''))
        payload_bytes += _wi32(len(blocks))
        for blk in blocks:
            label = blk['label']
            ch    = ch_map[label]
            payload_bytes += _encode_int16(
                np.asarray(blk['values'], dtype=np.float64),
                ch['phys_min'], ch['phys_max'],
            )

    # patch payload_offsets in index_bytes (sequential — just compute them)
    header_size  = len(hdr)
    index_size   = len(index_bytes)
    payload_base = header_size + index_size

    # re-build index with real offsets
    index_bytes = b""
    running_offset = payload_base
    for w in waves:
        blocks = w['blocks']
        n_blocks = len(blocks)
        index_bytes += (
            _wstr(w['annot'])
            + _wstr(w.get('instance', '.'))
            + _wstr(w.get('annot_ch', '.'))
            + _wf64(w.get('annot_start_sec', 0.0))
            + _wf64(w.get('annot_stop_sec',  0.0))
            + _wf64(w.get('anchor_sec',       0.0))
            + _wf64(w.get('wave_start_sec',  -1.5))
            + _wf64(w.get('wave_stop_sec',    1.5))
            + _wu64(running_offset)
            + _wi32(n_blocks)
        )
        payload_sz = 4 + len(w.get('meta', '').encode()) + 4  # meta str + n_blocks
        for blk in blocks:
            payload_sz += len(blk['values']) * 2
        for blk in blocks:
            ns = len(blk['values'])
            index_bytes += _wi32(ns) + _wf64(-1.5) + _wf64(1.5)
        running_offset += payload_sz

    path.write_bytes(hdr + index_bytes + payload_bytes)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR      = 128.0
N_SAMP  = 384          # 3 s at 128 Hz
W_LEFT  = N_SAMP // 2  # anchor at midpoint

def _sine_wave(freq: float = 1.0, amp: float = 50.0) -> np.ndarray:
    t = np.linspace(-1.5, 1.5, N_SAMP, endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)


@pytest.fixture
def two_channel_lwf(tmp_path) -> Path:
    """Standard mode: 4 waves, 2 channels each."""
    p = tmp_path / "standard.lwf"
    channels = [
        {'label': 'CZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
        {'label': 'FZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
    ]
    rng = np.random.default_rng(42)
    waves = []
    for i in range(4):
        waves.append({
            'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': '.',
            'anchor_sec': 100.0 + i * 10,
            'wave_start_sec': 100.0 + i * 10 - 1.5,
            'wave_stop_sec':  100.0 + i * 10 + 1.5,
            'meta': '',
            'blocks': [
                {'label': 'CZ', 'values': _sine_wave(1.0) + rng.normal(0, 5, N_SAMP)},
                {'label': 'FZ', 'values': _sine_wave(2.0) + rng.normal(0, 5, N_SAMP)},
            ],
        })
    make_lwf(p, channels=channels, waves=waves)
    return p


@pytest.fixture
def annot_ch_match_lwf(tmp_path) -> Path:
    """annot-ch-match mode: 6 waves, 1 block per wave (alternating CZ/FZ)."""
    p = tmp_path / "annot_ch_match.lwf"
    channels = [
        {'label': 'CZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
        {'label': 'FZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
    ]
    rng = np.random.default_rng(7)
    waves = []
    for i in range(6):
        ch = 'CZ' if i % 2 == 0 else 'FZ'
        waves.append({
            'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': ch,
            'anchor_sec': 100.0 + i * 10,
            'wave_start_sec': 100.0 + i * 10 - 1.5,
            'wave_stop_sec':  100.0 + i * 10 + 1.5,
            'meta': f'wave_{i}',
            'blocks': [
                {'label': ch, 'values': _sine_wave() + rng.normal(0, 5, N_SAMP)},
            ],
        })
    make_lwf(p, channels=channels, waves=waves, annot_ch_match=True)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReadHeader:
    def test_reads_phys_min_max(self, two_channel_lwf):
        with open(two_channel_lwf, 'rb') as f:
            h = _read_header(f)
        cz = next(c for c in h['channels'] if c['label'] == 'CZ')
        assert cz['phys_min'] == pytest.approx(-200.0)
        assert cz['phys_max'] == pytest.approx(200.0)

    def test_channel_count(self, two_channel_lwf):
        with open(two_channel_lwf, 'rb') as f:
            h = _read_header(f)
        assert len(h['channels']) == 2
        assert [c['label'] for c in h['channels']] == ['CZ', 'FZ']

    def test_bad_magic_raises(self, tmp_path):
        p = tmp_path / "bad.lwf"
        p.write_bytes(b"NOPE" + b"\x00" * 100)
        with open(p, 'rb') as f:
            with pytest.raises(ValueError, match="bad magic"):
                _read_header(f)

    def test_bad_version_raises(self, tmp_path):
        p = tmp_path / "badver.lwf"
        p.write_bytes(b"LWF1" + struct.pack('<i', 99) + b"\x00" * 100)
        with open(p, 'rb') as f:
            with pytest.raises(ValueError, match="version"):
                _read_header(f)


class TestLwfSummary:
    def test_returns_dataframe(self, two_channel_lwf):
        df = lwf_summary(two_channel_lwf)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_n_waves(self, two_channel_lwf):
        df = lwf_summary(two_channel_lwf)
        assert df['n_waves'].iloc[0] == 4

    def test_n_channels(self, two_channel_lwf):
        df = lwf_summary(two_channel_lwf)
        assert df['n_channels'].iloc[0] == 2

    def test_channel_labels(self, two_channel_lwf):
        df = lwf_summary(two_channel_lwf)
        assert df['channels'].iloc[0] == 'CZ,FZ'

    def test_directory_scan(self, tmp_path, two_channel_lwf, annot_ch_match_lwf):
        df = lwf_summary(tmp_path)
        assert len(df) == 2


class TestFromLwfStandard:
    def test_shape(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        assert aat.shape == (4, 2, N_SAMP)

    def test_axes(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        assert aat.axes == ['instance', 'channel', 'sample']

    def test_channel_meta(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        labels = list(aat.axis_meta['channel']['label'])
        assert labels == ['CZ', 'FZ']

    def test_instance_meta_columns(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        for col in ('annot', 'annot_ch', 'anchor_sec', 'wave_start_sec'):
            assert col in aat.axis_meta['instance'].columns

    def test_time_axis_range(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        t = aat.axis_index['sample']
        assert t[0]  == pytest.approx(-1.5, abs=0.02)
        assert t[-1] == pytest.approx( 1.5, abs=0.02)

    def test_not_annot_ch_match(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        assert aat.attrs['annot_ch_match'] is False

    def test_values_finite(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        assert np.all(np.isfinite(aat.data))

    def test_int16_decode_precision(self, tmp_path):
        """Round-trip encode→int16→decode should stay within 1 LSB."""
        phys_min, phys_max = -100.0, 100.0
        rng = np.random.default_rng(0)
        original = rng.uniform(phys_min, phys_max, N_SAMP)

        channels = [{'label': 'CZ', 'unit': 'uV', 'sr': SR,
                     'phys_min': phys_min, 'phys_max': phys_max}]
        waves = [{'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': '.',
                  'anchor_sec': 0.0, 'wave_start_sec': -1.5, 'wave_stop_sec': 1.5,
                  'meta': '', 'blocks': [{'label': 'CZ', 'values': original}]}]
        p = tmp_path / "precision.lwf"
        make_lwf(p, channels=channels, waves=waves)

        aat = from_lwf(p)
        recovered = aat.data[0, 0, :]
        lsb = (phys_max - phys_min) / 65535.0
        assert np.max(np.abs(recovered - original)) <= lsb + 1e-9

    def test_multi_file_concat(self, tmp_path):
        """Two files with same channels are concatenated along instance axis."""
        channels = [{'label': 'CZ', 'unit': 'uV', 'sr': SR,
                     'phys_min': -200.0, 'phys_max': 200.0}]
        rng = np.random.default_rng(1)
        for name, n in [("a.lwf", 3), ("b.lwf", 5)]:
            waves = [{'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': '.',
                      'anchor_sec': float(i), 'wave_start_sec': float(i) - 1.5,
                      'wave_stop_sec': float(i) + 1.5, 'meta': '',
                      'blocks': [{'label': 'CZ',
                                  'values': rng.normal(0, 50, N_SAMP)}]}
                     for i in range(n)]
            make_lwf(tmp_path / name, channels=channels, waves=waves)

        aat = from_lwf(tmp_path)
        assert aat.shape[0] == 8      # 3 + 5
        assert aat.shape[1] == 1
        assert aat.shape[2] == N_SAMP

    def test_channel_mismatch_raises(self, tmp_path):
        ch_a = [{'label': 'CZ', 'unit': 'uV', 'sr': SR,
                 'phys_min': -200.0, 'phys_max': 200.0}]
        ch_b = [{'label': 'FZ', 'unit': 'uV', 'sr': SR,
                 'phys_min': -200.0, 'phys_max': 200.0}]
        rng = np.random.default_rng(2)
        for name, ch in [("x.lwf", ch_a), ("y.lwf", ch_b)]:
            waves = [{'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': '.',
                      'anchor_sec': 0.0, 'wave_start_sec': -1.5, 'wave_stop_sec': 1.5,
                      'meta': '', 'blocks': [{'label': ch[0]['label'],
                                              'values': rng.normal(0, 50, N_SAMP)}]}]
            make_lwf(tmp_path / name, channels=ch, waves=waves)

        with pytest.raises(ValueError, match="channel mismatch"):
            from_lwf(tmp_path)


class TestFromLwfAnnotChMatch:
    def test_shape_one_block_per_wave(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf)
        assert aat.shape == (6, 1, N_SAMP)

    def test_annot_ch_match_flag(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf)
        assert aat.attrs['annot_ch_match'] is True

    def test_annot_ch_in_instance_meta(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf)
        ch_vals = list(aat.axis_meta['instance']['annot_ch'])
        assert ch_vals == ['CZ', 'FZ', 'CZ', 'FZ', 'CZ', 'FZ']

    def test_meta_preserved(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf)
        metas = list(aat.axis_meta['instance']['meta'])
        assert metas == [f'wave_{i}' for i in range(6)]

    def test_values_finite(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf)
        assert np.all(np.isfinite(aat.data))

    def test_decode_uses_correct_channel_phys(self, tmp_path):
        """Each channel uses its own phys_min/phys_max for decoding."""
        channels = [
            {'label': 'CZ', 'unit': 'uV', 'sr': SR,
             'phys_min': -100.0, 'phys_max': 100.0},
            {'label': 'FZ', 'unit': 'uV', 'sr': SR,
             'phys_min': -500.0, 'phys_max': 500.0},
        ]
        cz_vals = np.full(N_SAMP, 80.0)   # within CZ range
        fz_vals = np.full(N_SAMP, -400.0)  # outside CZ range, within FZ range

        waves = [
            {'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': 'CZ',
             'anchor_sec': 0.0, 'wave_start_sec': -1.5, 'wave_stop_sec': 1.5,
             'meta': '', 'blocks': [{'label': 'CZ', 'values': cz_vals}]},
            {'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': 'FZ',
             'anchor_sec': 10.0, 'wave_start_sec': 8.5, 'wave_stop_sec': 11.5,
             'meta': '', 'blocks': [{'label': 'FZ', 'values': fz_vals}]},
        ]
        p = tmp_path / "phys_check.lwf"
        make_lwf(p, channels=channels, waves=waves, annot_ch_match=True)

        aat = from_lwf(p)
        lsb_cz = 200.0 / 65535.0
        lsb_fz = 1000.0 / 65535.0
        assert np.max(np.abs(aat.data[0, 0, :] - 80.0))   <= lsb_cz + 1e-9
        assert np.max(np.abs(aat.data[1, 0, :] - (-400.0))) <= lsb_fz + 1e-9
