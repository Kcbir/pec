from fractions import Fraction

import pytest

from pec import pi_equivalent, separating_action_depth, separating_mass, witness_pair
from pec.envs.corridor import make_pair
from pec.envs.suite import SUITE
from pec.envs.tmaze import (
    ACTIONS, OBSERVATIONS, SAW_LEFT, blind_model, cue_model, mixed_policy, wait_policy,
)
from pec.rdp import Posterior, sample_trajectories

HALF = Fraction(1, 2)


def test_tmaze_is_equivalent_under_the_behaviour_policy():
    assert pi_equivalent(cue_model, blind_model, wait_policy)
    assert separating_action_depth(cue_model, blind_model, wait_policy) is None


def test_tmaze_witness_names_the_missing_experiment():
    assert witness_pair(cue_model, blind_model, mixed_policy(HALF)) == ((SAW_LEFT, None), "run")
    assert separating_action_depth(cue_model, blind_model, mixed_policy(HALF)) == 2


def test_posterior_odds_equal_prior_odds_under_the_behaviour_policy():
    data = sample_trajectories(cue_model, wait_policy, 50, 6, seed=0)
    assert Posterior([cue_model, blind_model]).odds(data) == 1


def test_separating_mass_is_zero_exactly_when_equivalent():
    prior = [HALF, HALF]
    models = [cue_model, blind_model]
    assert separating_mass(models, prior, wait_policy, 4, ACTIONS, OBSERVATIONS) == 0
    assert separating_mass(models, prior, mixed_policy(HALF), 4, ACTIONS, OBSERVATIONS) > 0


@pytest.mark.parametrize("env", SUITE, ids=lambda e: e["name"])
def test_suite_pairs_are_equivalent_until_the_deviation(env):
    assert pi_equivalent(env["true"], env["rival"], env["policy"])
    assert not pi_equivalent(env["true"], env["rival"], env["deviation"])


@pytest.mark.parametrize("horizon", [1, 2, 3, 5, 8])
def test_corridor_separating_horizon_is_the_parameter(horizon):
    true, rival, policy, deviation = make_pair(horizon)
    assert pi_equivalent(true, rival, policy)
    assert separating_action_depth(true, rival, deviation) == horizon
