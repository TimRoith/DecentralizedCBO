"""Download MNIST into nn_problems.DATA_ROOT.

Run once before any job that uses MNIST; on a cluster, run it on a login node,
since compute nodes usually have no outbound network.

    python fetch_data.py
"""
import sys


def main():
    from nn_problems import DATA_ROOT, _load

    print('data root: {}'.format(DATA_ROOT))
    for name in ('digits', 'mnist'):
        try:
            Xtr, Xte, ytr, yte = _load(name)
            print('  {:8s} ok   train {}  test {}'.format(
                name, Xtr.shape, Xte.shape))
        except Exception as e:
            print('  {:8s} FAILED: {}'.format(name, e))
            return 1

    raw = DATA_ROOT / 'MNIST' / 'raw'
    if raw.is_dir():
        files = sorted(p for p in raw.iterdir() if p.is_file())
        total = sum(p.stat().st_size for p in files)
        print('\n{} files in {} ({:.0f} MB)'.format(len(files), raw, total / 1e6))
        for p in files:
            print('  {:28s} {:8.1f} MB'.format(p.name, p.stat().st_size / 1e6))
    print('\nReady. Compute nodes will now read from disk without a download.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
