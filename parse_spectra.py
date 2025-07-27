#!/usr/bin/env python3
"""
parse_spectra.py

Parse Gaussian log files to extract excitation wavelengths and oscillator strengths,
compute a broadened spectrum across 0–800 nm, and save results to a CSV.

Usage:
  python parse_spectra.py [--start N] [--end M] [--width W] [--kind {gaussian,lorentzian}] [--csv FILE]

Options:
  --start N       First molecule index (inclusive). If omitted, all matching logs are scanned.
  --end M         Last molecule index (inclusive). Only used if --start is provided.
  --width W       Broadening width in nm (sigma for Gaussian or gamma for Lorentzian). Default: 10.0
  --kind K        Broadening type: 'gaussian' or 'lorentzian'. Default: 'gaussian'
  --csv FILE      Output CSV filename. Default: 'spectra.csv'

Each row in the CSV will contain:
  - molecule: integer index
  - wavelengths: JSON list of excitation wavelengths (nm)
  - oscillator_strengths: JSON list of oscillator strengths
  - width: the broadening width used
  - kind: broadening type
  - spectrum: JSON list of normalized intensities at 0–800 nm

Automatically checks only the last 10 lines of a log for "Normal termination" to ensure completion before parsing.
"""

import os
import re
import glob
import json
import argparse

import numpy as np
import pandas as pd

# Define the expected DataFrame columns
COLUMNS = [
    'molecule', 'wavelengths', 'oscillator_strengths',
    'width', 'kind', 'spectrum'
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse Gaussian log files and create broadened spectra."
    )
    parser.add_argument(
        '--start', type=int, help='Start molecule index (inclusive)'
    )
    parser.add_argument(
        '--end', type=int, help='End molecule index (inclusive)'
    )
    parser.add_argument(
        '--width', type=float, default=10.0,
        help='Broadening width in nm'
    )
    parser.add_argument(
        '--kind', choices=['gaussian','lorentzian'], default='gaussian',
        help='Broadening type'
    )
    parser.add_argument(
        '--csv', default='spectra.csv',
        help='Output CSV file'
    )
    return parser.parse_args()


def parse_excitation_data(logfile):
    """Extract (wavelength_nm, oscillator_strength) from a Gaussian log file."""
    pattern = re.compile(r'([0-9]+\.?[0-9]*) eV\s+([0-9]+\.?[0-9]*) nm\s+f=([0-9]+\.?[0-9]*)')
    excitations = []
    with open(logfile, 'r') as f:
        for line in f:
            if 'Excited State' in line:
                m = pattern.search(line)
                if m:
                    excitations.append((float(m.group(2)), float(m.group(3))))
    return excitations


def compute_spectrum(excitations, x, width, kind):
    """Compute broadened (normalized) spectrum for excitations over x-axis."""
    y = np.zeros_like(x, dtype=float)
    for lam, f in excitations:
        if kind == 'gaussian':
            y += f * np.exp(-((x - lam)**2) / (2 * width**2))
        else:  # lorentzian
            y += f * (width**2 / ((x - lam)**2 + width**2))
    # Normalize
    if y.max() > 0:
        y /= y.max()
    return y


def main():
    args = parse_args()
    
    # Determine molecule indices
    if args.start is not None:
        start = args.start
        end = args.end if args.end is not None else args.start
        mol_idxs = list(range(start, end+1))
    else:
        # scan all log files matching 'molecule_####.log'
        mol_idxs = []
        for fn in glob.glob('molecule_*.log'):
            m = re.match(r'molecule_(\d{4})\.log', fn)
            if m:
                mol_idxs.append(int(m.group(1)))
        mol_idxs.sort()

    # Load or initialize DataFrame
    if os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        existing = set(df['molecule'].astype(int))
    else:
        df = pd.DataFrame(columns=[
            'molecule', 'wavelengths', 'oscillator_strengths',
            'width', 'kind', 'spectrum'
        ])
        existing = set()

    # Prepare x-axis for spectrum 0–800 nm
    x = np.arange(0, 801, 1)

    records = []
    for idx in mol_idxs:
        if idx in existing:
            continue
        logfile = f"molecule_{idx:04d}.log"
        if not os.path.isfile(logfile):
            print(f"Warning: {logfile} not found, skipping.")
            continue

        # Automatic filtering: ensure job completed by checking last 10 lines
        with open(logfile, 'r') as f:
            lines = f.readlines()
        tail = ''.join(lines[-10:])
        if 'Normal termination' not in tail:
            print(f"Skipping {logfile}: no Normal termination in last 10 lines.")
            continue

        excitations = parse_excitation_data(logfile)
        if not excitations:
            print(f"No excitations found in {logfile}, skipping.")
            continue

        wavelengths = [e[0] for e in excitations]
        strengths = [e[1] for e in excitations]
        spectrum = compute_spectrum(excitations, x, args.width, args.kind)

        records.append({
            'molecule': idx,
            'wavelengths': json.dumps(wavelengths),
            'oscillator_strengths': json.dumps(strengths),
            'width': args.width,
            'kind': args.kind,
            'spectrum': json.dumps(spectrum.tolist())
        })

    if records:
        df_new = pd.DataFrame(records, columns=COLUMNS)
        df = pd.concat([df, df_new], ignore_index=True)
        df = df.reindex(columns=COLUMNS)
        df.sort_values('molecule', inplace=True)
        df.to_csv(args.csv, index=False)
        print(f"Appended {len(records)} records to {args.csv}")
    else:
        print("No new records to append.")


if __name__ == "__main__":
    main()