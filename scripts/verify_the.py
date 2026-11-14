"""Reproduces the data checks reported in the paper: for each edition, the weighted
sum of the pillars vs. the published overall score, and the agreement of FN-MABAC
(TFNs (0,100,100), published WUR 3.0 weights) with the official ranking.
Run from this folder: python3 verify_the.py"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["x", "../data/THE-World-University-Rankings/outputs/csv", "2025", "../results", "1"]
import numpy as np
import experiment as e

for y in range(2023, 2028):
    d = e.load(y)
    X = d[e.PILLARS].to_numpy(float)
    err = np.abs(X @ e.W_TRUE - d.Overall.to_numpy())
    r = e.rw(e.fn_mabac(X, e.W_TRUE, e.CORE_TRUE), e.rank_desc(d.Overall.to_numpy()))
    print(f"{y}: {len(d)} institutions with exact overall score, "
          f"max |weighted sum - overall| = {err.max():.3f}, rw(FN-MABAC, THE) = {r:.5f}")
