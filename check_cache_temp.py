import os, pickle, sys

os.chdir('c:/Users/BundaST/PycharmProjects/SAMOS-Project/SAMOS-NAS')

for suite_pid in [('c10mop', 8), ('c10mop', 9)]:
    suite, pid = suite_pid
    root = f'results/evoxbench/{suite}/pid{pid}/B1200_P20'

    # Check pareto_approx.pkl
    pa_path = os.path.join(root, 'pareto_approx.pkl')
    if os.path.isfile(pa_path):
        with open(pa_path, 'rb') as f:
            pa = pickle.load(f)
        nb = pa.get('norm_bounds')
        print(f'{suite}/pid{pid} pareto_approx norm_bounds: {nb is not None} ({nb})')
        print(f'  ref_point: {pa.get("ref_point")}')
        print(f'  pareto_approx shape: {pa["pareto_approx"].shape}')
    else:
        print(f'{suite}/pid{pid} no pareto_approx.pkl')

    # Check indicators_cache.pkl
    ic_path = os.path.join(root, 'indicators_cache.pkl')
    if os.path.isfile(ic_path):
        with open(ic_path, 'rb') as f:
            ic = pickle.load(f)
        methods = ic.get('methods', {})
        print(f'  indicators_cache approx_mtime: {ic.get("approx_mtime")}')
        import os as _os
        pa_mtime = _os.path.getmtime(pa_path) if _os.path.isfile(pa_path) else None
        print(f'  pareto_approx.pkl actual mtime: {pa_mtime}')
        print(f'  mtime match: {abs(ic.get("approx_mtime", -1) - (pa_mtime or 0)) < 1e-3}')
        for k, v in methods.items():
            hv = v.get('hv')
            if hv is not None:
                print(f'  {k}: mean_hv={hv.mean():.4f}, vals={hv[:3]}')
    else:
        print(f'  no indicators_cache.pkl')
    print()
