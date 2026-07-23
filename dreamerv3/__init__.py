"""Forked DreamerV3 engine (from github.com/NM512/dreamerv3-torch).

Local edits vs upstream (the code below is the source of truth):
  - tools.py: OneHotDist numerical stability (clamped/nan-guarded logits,
    unimix in probability space) for large discrete action spaces;
    DiscDist buckets on logits.device instead of a hardcoded device;
    sample_episodes / cache handling made numpy-2.x-safe (no ragged
    np.append; list-vs-array episode normalization).
  - networks.py: MLP default device derived from parameters, not hardcoded.
  - models.py: optional image preprocessing (uint8 -> float only when
    needed); mask head + predicted-mask logit suppression in imagination and
    ground-truth masking in the real env (faithful action masking, P0.3;
    see ImagBehavior._policy_dist / _apply_mask).
  - dreamer.py / exploration.py / parallel.py: unchanged from upstream.
"""
