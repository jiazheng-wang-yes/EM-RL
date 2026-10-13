"""Exercise upstream virtual filesystem detection without running an agent."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.source / 'hvta/FilesystemWrapper/virtual_fs.py'
    spec = importlib.util.spec_from_file_location('audited_hve_fs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fs = module.VirtualFS()
    original = fs.populate_hidden_solution('AUDIT_SECRET_NOT_A_REAL_SOLUTION')
    direct = f'[fs_cat {original}]'
    direct_result = fs.dispatch(direct)
    direct_flag = fs.is_cat_of(direct, original) and not direct_result.startswith('cat:')
    rename = fs.dispatch(f'[fs_mv {original} /audit_moved.txt]')
    moved = '[fs_cat /audit_moved.txt]'
    moved_result = fs.dispatch(moved)
    moved_flag = fs.is_cat_of(moved, original) and not moved_result.startswith('cat:')
    result = {
        'source_commit': subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
        'virtual_fs_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'direct_read_returns_secret': direct_result == 'AUDIT_SECRET_NOT_A_REAL_SOLUTION',
        'direct_read_matches_wrapper_detector': direct_flag,
        'rename_result': rename,
        'renamed_read_returns_secret': moved_result == 'AUDIT_SECRET_NOT_A_REAL_SOLUTION',
        'renamed_read_matches_wrapper_detector': moved_flag,
        'scope': 'Executes upstream VirtualFilesystem and reproduces the path predicate used by handle_fs_step. No full TextArena episode or training run.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
