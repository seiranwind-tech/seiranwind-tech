import sys, os, hashlib, importlib.util, numpy as np
ENG_DIR = os.environ["F50_ENGINE_DIR"]
assert hashlib.sha256(open(os.path.join(ENG_DIR, "f40_v39r3_engine.py"), "rb").read()).hexdigest() == "4a71aeb4a66a8a071a6b6f9a9b065687d237909aea5e89a537ee6d80d2aa3b1d"
sys.path.insert(0, ENG_DIR); import f40_v39r3_engine as E
_W9 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w9_full_gate_closure", "w9_shadow.py")
assert hashlib.sha256(open(_W9, "rb").read()).hexdigest() == "07ef512ce4a4bb54668bc42fc1082275d3c0678265dfff057a8d4d76aba27b00"
spec = importlib.util.spec_from_file_location("w9_shadow", _W9); W9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(W9)
nr = E.normalize_rows
make_engine = W9.make_engine; observed = W9.observed; Shadow = W9.Shadow

def decompose(eng):
    """Exact per-well force split of one engine zin update (reconstructed from engine methods, no mutation)."""
    c = eng.config; z = eng.zin; zout = eng.zout
    zo1 = zout + c.dt * (0.3 * eng._implicit_blade_action(zout) + eng._fsalt(zout))
    nrm = np.linalg.norm(zo1, axis=1, keepdims=True); zo1 = np.where(nrm > c.wall, zo1 * c.wall / nrm, zo1)
    wave = 0.6 * (zo1 - eng.zop) / max(c.wall, 1e-9)
    inner = eng._fsalt(z); salt = 0.5 * (inner - z * np.sum(z * inner, 1, keepdims=True))
    attr = nr(eng._attraction(z))
    zplus = nr(z + c.dt * (salt + wave + c.attract_gain * attr))
    return salt, wave, attr, zplus
