#!/usr/bin/env python3
"""
parse_spectra.py

Parse Gaussian log files to extract excitation wavelengths and oscillator strengths,
compute a broadened spectrum across 0–800 nm, extract optimized geometry (with element symbols), and save results to a CSV.

Usage:
  python parse_spectra.py [--start N] [--end M] [--width W] [--kind {gaussian,lorentzian}] [--csv FILE]
"""
import os
import re
import glob
import json
import argparse
import warnings
import numpy as np
import pandas as pd

# Suppress specific FutureWarning from pandas concat
warnings.filterwarnings(
    "ignore",
    message=".*DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning
)

# Periodic table mapping: atomic number -> symbol (1-36)
PERIODIC_TABLE = {
    1: 'H',  2: 'He', 3: 'Li', 4: 'Be', 5: 'B',  6: 'C',  7: 'N',  8: 'O',
    9: 'F', 10: 'Ne',11: 'Na',12: 'Mg',13: 'Al',14: 'Si',15: 'P', 16: 'S',
   17: 'Cl',18: 'Ar',19: 'K', 20: 'Ca',21: 'Sc',22: 'Ti',23: 'V', 24: 'Cr',
   25: 'Mn',26: 'Fe',27: 'Co',28: 'Ni',29: 'Cu',30: 'Zn',31: 'Ga',32: 'Ge',
   33: 'As',34: 'Se',35: 'Br',36: 'Kr'
}

# Define expected DataFrame columns
COLUMNS = [
    'molecule', 'wavelengths', 'oscillator_strengths',
    'width', 'kind', 'spectrum', 'geometry'
]

# Regex to detect atomic data lines
ATOM_LINE = re.compile(r'^\s*(\d+)\s+(\d+)')


def parse_args():
    p = argparse.ArgumentParser(description="Parse Gaussian logs, spectra, and geometries.")
    p.add_argument('--start', type=int, help='Start molecule index')
    p.add_argument('--end', type=int, help='End molecule index')
    p.add_argument('--width', type=float, default=10.0, help='Spectral broadening width (nm)')
    p.add_argument('--kind', choices=['gaussian','lorentzian'], default='gaussian', help='Broadening kernel')
    p.add_argument('--csv', default='spectra.csv', help='Output CSV file')
    return p.parse_args()


def parse_excitation_data(logfile):
    """Extract (wavelength_nm, oscillator_strength) from Gaussian log."""
    pattern = re.compile(r'([0-9]+\.?[0-9]*) eV\s+([0-9]+\.?[0-9]*) nm\s+f=([0-9]+\.?[0-9]*)')
    excitations = []
    with open(logfile) as f:
        for line in f:
            if 'Excited State' in line:
                m = pattern.search(line)
                if m:
                    excitations.append((float(m.group(2)), float(m.group(3))))
    return excitations


def parse_optimized_geometry(logfile):
    """Extract last 'Standard orientation' block, convert atomic numbers to symbols."""
    last_block = []
    with open(logfile) as f:
        current = None
        for line in f:
            if 'Standard orientation' in line:
                current = []
                for _ in range(4): next(f, None)
                continue
            if current is None:
                continue
            if line.strip().startswith('----'):
                last_block = current
                break
            if not ATOM_LINE.match(line):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            atomic_num = int(parts[1])
            symbol = PERIODIC_TABLE.get(atomic_num, 'X')
            x, y, z = parts[3], parts[4], parts[5]
            current.append(f"{symbol} {x} {y} {z}")
    return '\n'.join(last_block)


def compute_spectrum(ex, x, width, kind):
    y = np.zeros_like(x, dtype=float)
    for lam, f in ex:
        if kind == 'gaussian':
            y += f * np.exp(-((x-lam)**2)/(2*width**2))
        else:
            y += f * (width**2/((x-lam)**2 + width**2))
    if y.max() > 0: y /= y.max()
    return y


def main():
    args = parse_args()
    if args.start is not None:
        idxs = range(args.start, (args.end or args.start) + 1)
    else:
        idxs = sorted(int(re.match(r'molecule_(\d{4})\.log', fn).group(1)) for fn in glob.glob('molecule_*.log'))

    if os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        df = df.reindex(columns=COLUMNS)
        existing = set(df['molecule'].astype(int))
    else:
        df = pd.DataFrame(columns=COLUMNS)
        existing = set()

    x = np.arange(0, 801, 1)
    records = []
    for idx in idxs:
        if idx in existing: continue
        logfile = f"molecule_{idx:04d}.log"
        if not os.path.isfile(logfile):
            print(f"Warning: {logfile} not found, skipping.")
            continue
        lines = open(logfile).read().splitlines()
        if 'Normal termination' not in '\n'.join(lines[-10:]):
            print(f"Skipping {logfile}: not terminated normally.")
            continue
        ex = parse_excitation_data(logfile)
        if not ex:
            print(f"No excitations in {logfile}, skipping.")
            continue
        geom = parse_optimized_geometry(logfile)
        lam = [e[0] for e in ex]
        strg = [e[1] for e in ex]
        spec = compute_spectrum(ex, x, args.width, args.kind)
        records.append({'molecule': idx,
                        'wavelengths': json.dumps(lam),
                        'oscillator_strengths': json.dumps(strg),
                        'width': args.width,
                        'kind': args.kind,
                        'spectrum': json.dumps(spec.tolist()),
                        'geometry': geom})
    if records:
        df2 = pd.DataFrame(records, columns=COLUMNS)
        # perform concat and then reindex to ensure consistent columns
        df = pd.concat([df, df2], ignore_index=True, sort=False)
        df = df.reindex(columns=COLUMNS)
        df.sort_values('molecule', inplace=True)
        df.to_csv(args.csv, index=False)
        print(f"Appended {len(records)} records to {args.csv}")
    else:
        print("No new records to append.")

if __name__ == '__main__':
    main()

