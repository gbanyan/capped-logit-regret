"""Verify a clean saved-result replay, without neural training or inference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['path']
        assert path.is_file() and sha(path) == item['sha256'], item['path']
    folder = Path(tempfile.mkdtemp(prefix='capped-logit-reproduce-'))
    work = folder / 'resource'
    shutil.copytree(ROOT / 'resource', work)
    env = dict(os.environ, MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='2',
               OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    commands = []

    def run(arguments):
        result = subprocess.run([sys.executable, *arguments], cwd=work, env=env,
                                capture_output=True, text=True)
        commands.append({'arguments': arguments, 'returncode': result.returncode})
        if result.returncode:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError('Replay failed: ' + arguments[0])

    run(['-I', '-S', str(work / 'theory_checks/verify_counterexample.py'),
         '--output', str(work / 'theory_result.json')])
    theory = json.loads((work / 'theory_result.json').read_text())
    expected = json.loads((work / 'theory_checks/expected_result.json').read_text())
    actual = dict(theory)
    actual.pop('python')
    expected.pop('python')
    assert actual == expected, 'Theory result differs beyond Python version'
    run([str(work / 'regenerate.py')])
    generated = json.loads((work / 'regeneration_check.json').read_text())
    assert generated['status'] == 'PASS' and generated['outputs'] == 17
    names = ['natural_seed_dispersion.csv', 'paired_natural_error.csv',
             'affine_seed_dispersion.csv']
    before = {n: sha(work / 'diagnostics' / n) for n in names}
    run([str(work / 'producer_sources/new_diagnostic_extraction/reporting_diagnostics.py'),
         '--resource-root', str(work)])
    assert all(sha(work / 'diagnostics' / n) == value for n, value in before.items())
    import numpy
    import matplotlib
    report = {'status': 'PASS', 'release': manifest['version'],
              'files_verified': len(manifest['files']), 'python': sys.version,
              'numpy': numpy.__version__, 'matplotlib': matplotlib.__version__,
              'theory_status': theory['status'], 'numerical_samples':
              theory['numerical_sanity']['samples'], 'saved_outputs': 17,
              'descriptive_csvs': 3, 'output_directory': str(work),
              'commands': commands,
              'scope': 'Fresh saved-result replay; no neural training, inference, or interval-solver replay'}
    target = ROOT / 'verification'
    target.mkdir(exist_ok=True)
    (target / 'last_run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
