"""Start the local Studio and validate its actual evidence bundle."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Invalid port')
    repo = Path(__file__).resolve().parents[1]
    root = f'http://127.0.0.1:{args.port}/'

    def probe():
        with urllib.request.urlopen(root+'data/bundle.json', timeout=2) as response:
            bundle = json.load(response)
        if bundle.get('model', {}).get('repo') != 'vllm-sr/d3-lite':
            raise RuntimeError('Port serves a different application; no process was stopped')
        with urllib.request.urlopen(root+'app.js', timeout=2) as response:
            if b'researchResults' not in response.read():
                raise RuntimeError('Studio JavaScript is missing')
        return bundle

    try:
        bundle = probe()
    except OSError:
        logs = Path('/workspace/.cloud-setup')
        logs.mkdir(parents=True, exist_ok=True)
        with (logs/'studio.log').open('ab', buffering=0) as output:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'http.server', str(args.port),
                                        '--bind', '127.0.0.1', '--directory', str(repo/'studio')],
                                       stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
        (logs/'studio.pid').write_text(str(process.pid)+'\n')
        for attempt in range(40):
            try:
                bundle = probe()
                break
            except OSError:
                if process.poll() is not None or attempt == 39:
                    raise RuntimeError('Studio failed to start; inspect /workspace/.cloud-setup/studio.log')
                time.sleep(.1)
    print('Studio HTML assets and evidence validated;', len(bundle.get('warehouse_v2', {}).get('core_models', [])),
          'v2 model/split rows; no external publication performed')


if __name__ == '__main__':
    main()
