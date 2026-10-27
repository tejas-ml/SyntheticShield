from syntheticshield.data import make_private_dataset
from syntheticshield.generators import build_generator
from syntheticshield.auditor import membership_inference_audit, canary_exposure_audit
from syntheticshield.metrics import synthetic_utility


def test_end_to_end_smoke():
    ds = make_private_dataset(seed=1, n_train=80, n_test=80, n_features=6, n_canaries_per_rarity=2, rarity_levels=[1.5, 2.5])
    gen = build_generator("no_defense", seed=2).fit(ds.train, ds.feature_columns, ds.label_column)
    syn = gen.sample(120)
    utility = synthetic_utility(syn, ds.test, ds.feature_columns, ds.label_column)
    audit = membership_inference_audit(ds.train, ds.test, syn, ds.feature_columns)
    canaries = canary_exposure_audit(ds.canaries, syn, ds.feature_columns)
    assert 0.0 <= utility["utility_auc"] <= 1.0
    assert 0.0 <= audit["membership_auc"] <= 1.0
    assert len(canaries) == len(ds.canaries)
