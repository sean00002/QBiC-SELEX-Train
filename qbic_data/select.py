"""Choose experiments from Table S1 by ID, TF, source, model list, or all."""

from .manifest import load_manifest, parse_pair_id


def select_experiments(manifest=None, exp_ids=(), tfs=(), sources=(), model_ids=(), all_experiments=False):
    """Experiment IDs that have their own selection cycles (cycle-zero libraries excluded)."""
    m = load_manifest(manifest)
    own = m[m.record_class != "cycle-zero"]
    chosen = set(exp_ids)
    chosen |= {parse_pair_id(mid)[0] for mid in model_ids}
    if tfs:
        chosen |= set(own[own.tf.isin(tfs) | own.source_tf_label.isin(tfs)].exp_id)
    if sources:
        chosen |= set(own[own.source.isin(sources)].exp_id)
    if all_experiments:
        chosen |= set(own.exp_id)
    unknown = chosen - set(m.exp_id)
    if unknown:
        raise KeyError(f"not in Table S1: {sorted(unknown)[:5]}")
    return sorted(chosen)

def read_id_list(path) -> list[str]:
    """One ID per line (a path to a model file also works: the name is used)."""
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line.split("/")[-1].removesuffix(".weights.qbic"))
    return ids
