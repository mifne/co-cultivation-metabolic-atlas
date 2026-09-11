from types import SimpleNamespace

import cobra
import pytest

import main


def test_audited_cli_options_select_explicit_physics_and_observations():
    options = main.cultivation_options(SimpleNamespace(dynamics='audited', control_dt=1.,
                                                       consortium_profile='pf-helper3'))
    assert options['fba_mode'] == 'separate'
    assert options['solver_backend'] == 'highs'
    assert options['ph_control_target'] == 7.
    assert options['observation_schema'] == 'metabolic_v2'
    assert options['consortium_profile'] == 'pf-helper3'


@pytest.mark.parametrize('extra', [dict(fba_mode='cooperative'), dict(solver_backend='glpk')])
def test_audited_cli_rejects_incompatible_solver_paths(extra):
    with pytest.raises(ValueError, match='Audited dynamics require'):
        main.cultivation_options(SimpleNamespace(dynamics='audited', **extra))


def test_factory_does_not_pass_simulator_only_options_to_environment(monkeypatch):
    seen = {}
    class CaptureSimulator:
        def __init__(self, **kwargs):
            seen['simulator'] = kwargs
    def capture_env(*, simulator, max_time):
        seen['env'] = dict(simulator=simulator, max_time=max_time)
        return simulator
    monkeypatch.setattr(main, 'dFBASimulator', CaptureSimulator)
    monkeypatch.setattr(main, 'ConsortiumEnv', capture_env)
    factory = main.make_env(None, dict(max_time=1.,
        cooperative_surrogate_require_qualified=False,
        cooperative_surrogate_validation_manifest='a-manifest.json'),
        preloaded_models={'NS21': cobra.Model('NS21')})
    factory()
    assert seen['simulator']['cooperative_surrogate_require_qualified'] is False
    assert seen['simulator']['cooperative_surrogate_validation_manifest'] == 'a-manifest.json'
    assert seen['env']['max_time'] == 1.


def test_requested_default_profile_reads_real_models_instead_of_unrelated_mock(monkeypatch):
    names = ['Actinoplanes_sp_OR16_lcp', 'Rhizobacter_gummiphilus_NS21',
             'Propionibacterium_freudenreichii_shermanii']
    models = {name: cobra.Model(name) for name in names}
    seen = []
    def load(path):
        seen.append(path)
        return models
    monkeypatch.setattr(main, 'load_sbml_models', load)
    selected = main.load_requested_models(None, 'pf-helper3')
    assert list(selected) == names
    assert str(seen[0]).endswith('models/sbml/final_consortium')
