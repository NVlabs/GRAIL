"""Adapt the vendored PPO train-loop iterator to an absolute update counter.

The vendored trainer starts its loop from 1 even after restoring ``global_step``. This
adapter compiles the same ``train`` method with only the outer iterator changed to
``range(restored_global_step + 1, total + 1)``, so a resumed run continues to the same
absolute target instead of training ``total`` more updates. A fresh run (``global_step=0``)
iterates exactly as upstream. No vendored file is written.
"""
import ast
from copy import deepcopy
import functools
import hashlib
from pathlib import Path

# SHA-256 of imports/SONIC-manipulation/gear_sonic/trl/trainer/ppo_trainer.py. Update it together
# with any change to that file after re-checking the loop shape below.
PINNED_TRAINER_SHA256 = 'dfbc7c0f71cfc98520faab4e8cafc8d7876b2d393fd631a97d37e33d4757380c'


_TARGET = {'total': None}


def absolute_iterator(start, total):
    expected = _TARGET['total']
    if type(start) is not int or type(total) is not int or start < 0 or start > total or total != expected:
        raise ValueError('Absolute PPO counter/target differs')
    return range(start + 1, total + 1)


def adapted_train_ast(source):
    tree = ast.parse(source)
    owners = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TRLPPOTrainer']
    if len(owners) != 1:
        raise ValueError('Vendored trainer class missing/ambiguous')
    methods = [n for n in owners[0].body if isinstance(n,ast.FunctionDef) and n.name == 'train']
    if len(methods) != 1 or methods[0].decorator_list:
        raise ValueError('Vendored train method missing/changed decorators')
    node = deepcopy(methods[0])
    expected = ast.dump(ast.parse('range(1, args.num_total_batches + 1)',mode='eval').body)
    matches = [n for n in ast.walk(node) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
               and n.target.id == 'batch_idx' and ast.dump(n.iter) == expected]
    if len(matches) != 1:
        raise ValueError('Vendored train absolute-loop shape changed')
    matches[0].iter = ast.copy_location(ast.parse('_contactfirst_absolute_iterator(int(self.state.global_step), args.num_total_batches)',mode='eval').body,matches[0].iter)
    return ast.fix_missing_locations(node)


def install(module, total):
    path = Path(module.__file__)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINNED_TRAINER_SHA256:
        raise ValueError('Vendored PPO trainer SHA differs; absolute iterator was not installed')
    if type(total) is not int or total < 1:
        raise ValueError('Absolute PPO target must be a positive integer')
    _TARGET['total'] = total
    original = module.TRLPPOTrainer.train
    node = adapted_train_ast(raw.decode())
    namespace = {}
    module.__dict__['_contactfirst_absolute_iterator'] = absolute_iterator
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),module.__dict__,namespace)
    adapted = namespace['train']
    functools.update_wrapper(adapted,original)
    module.TRLPPOTrainer.train = adapted
    module._contactfirst_absolute_target_installed = {'source_sha256':PINNED_TRAINER_SHA256,'total':total,
        'only_ast_change':'range(restored_global_step+1, absolute_total+1)','original_source_modified':False}
