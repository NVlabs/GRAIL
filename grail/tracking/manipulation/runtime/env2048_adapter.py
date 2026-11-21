"""Training-process adapters installed before the vendored SONIC modules are imported.

* ``patch_environment`` applies the run's explicit PhysX GPU buffer capacities before
  simulation creation and records the live values.
* ``patch_trainer`` installs the absolute PPO update counter and, for an explicit resume of
  the same run, restores each saved optimizer parameter group's learning rate (the vendored
  full-resume path would otherwise flatten them) and verifies the restored state exactly.

The vendored source and the single-environment evaluation child stay unchanged.
"""
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys
from functools import wraps

BUFFER_FIELDS = {
    'gpu_max_rigid_contact_count', 'gpu_max_rigid_patch_count',
    'gpu_found_lost_pairs_capacity', 'gpu_found_lost_aggregate_pairs_capacity',
    'gpu_total_aggregate_pairs_capacity', 'gpu_collision_stack_size',
    'gpu_heap_capacity', 'gpu_temp_buffer_capacity',
}

def validate_buffers(config):
    values = config.get('physx_buffers')
    if not isinstance(values, dict) or set(values) != BUFFER_FIELDS:
        raise ValueError('Explicit complete reviewed PhysX capacity dictionary required')
    for name, value in values.items():
        if type(value) is not int or value < 2**20 or value > (3758096384 if name == 'gpu_collision_stack_size' else 2**31):
            raise ValueError('PhysX capacity outside bounded launch contract: ' + name)
    if type(config.get('num_envs')) is not int or config['num_envs'] < 4 or config['num_envs'] % 4:
        raise ValueError('num_envs must be a positive multiple of the four PPO minibatches')
    return values

def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    import hashlib
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _equal(expected, actual):
    import torch
    if isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and expected.shape == actual.shape and expected.dtype == actual.dtype
                and torch.equal(expected.cpu(), actual.detach().cpu()))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and expected.keys() == actual.keys() and all(_equal(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, type(expected)) and len(expected) == len(actual) and all(_equal(a, b) for a, b in zip(expected, actual))
    return expected == actual


def verify_live_resume(config, checkpoint, trainer):
    """Check that actor, critic, optimizer and scheduler equal the run's own saved checkpoint."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    _require(int(trainer.state.global_step) == config['resume_from_step'], 'Restored trainer counter differs')
    _require(_equal(checkpoint['policy_state_dict'], model.policy.state_dict()), 'Restored actor differs')
    _require(model.value_model is not None and _equal(checkpoint['value_state_dict'], model.value_model.state_dict()),
             'Restored critic differs')
    _require(_equal(checkpoint['optimizer_state_dict'], trainer.optimizer.state_dict()), 'Restored optimizer differs')
    _require(_equal(checkpoint['lr_scheduler_state_dict'], trainer.lr_scheduler.state_dict()), 'Restored scheduler differs')
    return {'schema': 'contactfirst.resume_initialization.v1', 'status': 'verified_before_next_rollout',
            'checkpoint': str(Path(config['output_root']) / 'train/last.pt'),
            'checkpoint_sha256': config['resume_checkpoint_sha256'], 'global_step': config['resume_from_step'],
            'target': config['iterations'], 'exact_actor': True, 'exact_critic': True,
            'exact_optimizer': True, 'exact_scheduler': True,
            'optimizer_group_lrs': [float(g['lr']) for g in trainer.optimizer.param_groups]}

def patch_environment(module, config):
    cls = module.ModularTrackingEnvCfg
    original = cls.__init__
    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        values = validate_buffers(config)
        before = {}
        for name, value in values.items():
            if not hasattr(self.sim.physx, name):
                raise RuntimeError('Pinned IsaacLab does not support PhysX field: ' + name)
            before[name] = getattr(self.sim.physx, name)
            setattr(self.sim.physx, name, value)
        actual = {name: getattr(self.sim.physx, name) for name in values}
        if actual != values or self.scene.num_envs != config['num_envs']:
            raise RuntimeError('Actual training scene or PhysX capacities differ from run configuration')
        _write_once(Path(config['output_root']) / config['runtime_attempt'] / 'physx_buffers_live.json',
                    {'schema': 'contactfirst.env2048.physx_live.v1', 'num_envs': self.scene.num_envs,
                     'before': before, 'configured': values, 'actual': actual,
                     'source_checkout_modified': False, 'applied_before_simulation_creation': True})
        print('[physx_buffers] validated configured capacities before simulation creation', flush=True)
    cls.__init__ = initialize

def patch_trainer(module, config):
    """Absolute PPO counter; on resume keep each saved param group's LR and verify the restore."""
    from absolute_ppo_target import install as install_absolute_target
    install_absolute_target(module, int(config['iterations']))
    _write_once(Path(config['output_root']) / config['runtime_attempt'] / 'absolute_ppo_target_live.json', module._contactfirst_absolute_target_installed)
    if not config.get('resume'):
        return
    cls = module.TRLPPOTrainer
    original = cls.load_checkpoint
    @wraps(original)
    def load(self, checkpoint_path, resume=False, warm_resume=False, flexible_load=False):
        checkpoint_path = Path(checkpoint_path)
        expected = Path(config['output_root']) / 'train/last.pt'
        _require(checkpoint_path == expected, 'Resume may load only this run\'s own train/last.pt')
        _require(resume and not warm_resume and not flexible_load, 'Full exact resume is required')
        _require(_sha(checkpoint_path) == config['resume_checkpoint_sha256'], 'Resume checkpoint changed')
        checkpoint = original(self, str(checkpoint_path), resume=True, warm_resume=False, flexible_load=False)
        saved_groups = checkpoint['optimizer_state_dict']['param_groups']
        _require(len(saved_groups) == len(self.optimizer.param_groups), 'Optimizer group count differs')
        for saved, actual in zip(saved_groups, self.optimizer.param_groups):
            actual['lr'] = saved['lr']
        receipt = verify_live_resume(config, checkpoint, self)
        _write_once(Path(config['output_root']) / config['runtime_attempt'] / 'resume_initialization.json', receipt)
        print('[resume_initialization] exact actor/value/optimizer/scheduler and counters verified', flush=True)
        return checkpoint
    cls.load_checkpoint = load

def install(config):
    actions = {'gear_sonic.envs.manager_env.modular_tracking_env_cfg': patch_environment,
               'gear_sonic.trl.trainer.ppo_trainer': patch_trainer}
    class Loader(importlib.abc.Loader):
        def __init__(self, delegate, action, finder):
            self.delegate, self.action, self.finder = delegate, action, finder
        def create_module(self, spec):
            return self.delegate.create_module(spec)
        def exec_module(self, module):
            self.delegate.exec_module(module)
            self.action(module, config)
            self.finder.pending.pop(module.__name__)
            if not self.finder.pending:
                sys.meta_path.remove(self.finder)
        def __getattr__(self, name):
            return getattr(self.delegate, name)
    class Finder(importlib.abc.MetaPathFinder):
        pending = dict(actions)
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in self.pending:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                raise ImportError('Pinned module unavailable: ' + fullname)
            spec.loader = Loader(spec.loader, self.pending[fullname], self)
            return spec
    for name in actions:
        if name in sys.modules:
            raise RuntimeError('Runtime adapter must be installed before pinned modules are imported')
    sys.meta_path.insert(0, Finder())
