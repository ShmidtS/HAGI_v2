"""Заглушка torch.distributed в памяти + импорт transformers-эталона."""
import sys, types
import torch

# --- пакет-заглушка torch.distributed ---
dist_pkg = types.ModuleType('torch.distributed')
dist_pkg.__path__ = []
dist_pkg.is_available = lambda: True
dist_pkg.is_initialized = lambda: False
dist_pkg.get_rank = lambda: 0
dist_pkg.get_world_size = lambda: 1
dist_pkg.init_process_group = lambda *a, **k: None
dist_pkg.destroy_process_group = lambda *a, **k: None
dist_pkg.barrier = lambda *a, **k: None
dist_pkg.reduce_scatter_tensor = lambda *a, **k: None
dist_pkg.all_gather_tensor = lambda *a, **k: None
dist_pkg.all_reduce = lambda *a, **k: None
dist_pkg.all_gather = lambda *a, **k: None
dist_pkg.broadcast = lambda *a, **k: None
dist_pkg.reduce = lambda *a, **k: None
dist_pkg.new_group = lambda *a, **k: None
dist_pkg.get_process_group_ranks = lambda *a, **k: []
dist_pkg.ReduceOp = type('ReduceOp', (), {'SUM': 0, 'AVG': 1, 'PRODUCT': 2, 'MIN': 3, 'MAX': 4})
dist_pkg.Backend = type('Backend', (), {'GLOO': 'gloo', 'NCCL': 'nccl', 'register_backend': staticmethod(lambda *a, **k: None)})
dist_pkg.ProcessGroup = type('ProcessGroup', (), {})
dist_pkg.Store = type('Store', (), {})  # нужно torch/testing fake_pg (torch.compile)
dist_pkg.gather = lambda *a, **k: None
dist_pkg.scatter = lambda *a, **k: None
sys.modules['torch.distributed'] = dist_pkg
torch.distributed = dist_pkg

def _bind_submodule(parent_name, child_name, mod):
    """Зарегистрировать подмодуль в sys.modules и повесить атрибут на родителя
    (dynamo иногда обращается по атрибуту, а не по import)."""
    full = f'{parent_name}.{child_name}'
    sys.modules[full] = mod
    parent = sys.modules[parent_name]
    setattr(parent, child_name, mod)
    return mod

# подмодули
tensor_pkg = types.ModuleType('torch.distributed.tensor')
tensor_pkg.__path__ = []
class _DTensor:
    def __init__(self, *a, **k): pass
    @staticmethod
    def from_local(*a, **k): raise NotImplementedError
    def to_local(self): return None
    def __repr__(self): return "DTensor(stub)"
tensor_pkg.DTensor = _DTensor
_bind_submodule('torch.distributed', 'tensor', tensor_pkg)

pt = types.ModuleType('torch.distributed.tensor.placement_types')
pt.Replicate = type('Replicate', (), {})
pt.Shard = type('Shard', (), {})
pt.Partial = type('Partial', (), {})
pt.Placement = type('Placement', (), {})
pt._StridedShard = type('_StridedShard', (), {})
pt.TensorMeta = type('TensorMeta', (), {})
_bind_submodule('torch.distributed.tensor', 'placement_types', pt)

utils = types.ModuleType('torch.distributed.tensor._utils')
utils.compute_local_shape_and_global_offset = lambda *a, **k: (None, None)
_bind_submodule('torch.distributed.tensor', '_utils', utils)

funcol = types.ModuleType('torch.distributed._functional_collectives')
funcol.all_gather_tensor = lambda *a, **k: None
funcol.traceable_collective_remaps = {}  # dynamo functions.py
_bind_submodule('torch.distributed', '_functional_collectives', funcol)

c10d = types.ModuleType('torch.distributed.distributed_c10d')
c10d._get_default_group = lambda: None
c10d.get_rank = lambda: 0
c10d.get_world_size = lambda: 1
# dynamo (WorldMetaClassVariable) импортирует _WorldMeta для проверки типа;
# без него ломается torch.compile.
c10d._WorldMeta = type('_WorldMeta', (), {})
# dynamo может дергать и эти имена (torch/distributed/distributed_c10d.py)
c10d._world = type('_WorldMeta_obj', (), {'pg': None, 'ranks': range(1), 'size': 1})()
# dynamo is_constant_pg_functions импортирует эти имена — нужны для torch.compile
c10d._get_group_size_by_name = lambda *a, **k: 1
c10d._get_group_tag = lambda *a, **k: ''
c10d._rank_not_in_group = lambda *a, **k: False
c10d._resolve_group_name_by_ranks_and_tag = lambda *a, **k: ''
c10d.get_process_group_ranks = lambda *a, **k: [0]
c10d.ProcessGroup = type('ProcessGroup', (), {})
_bind_submodule('torch.distributed', 'distributed_c10d', c10d)

# нативный torch._C._distributed_c10d отсутствует в этой ROCm-сборке;
# dynamo (ProcessGroupVariable) делает from torch._C._distributed_c10d import ProcessGroup.
# from-import по полному имени сначала смотрит в sys.modules — подкладываем заглушку.
c10d_native = types.ModuleType('torch._C._distributed_c10d')
c10d_native.ProcessGroup = type('ProcessGroup', (), {})
c10d_native.FakeProcessGroup = type('FakeProcessGroup', (c10d_native.ProcessGroup,), {})
sys.modules['torch._C._distributed_c10d'] = c10d_native

# device_mesh тоже может понадобиться
dm = types.ModuleType('torch.distributed.device_mesh')
dm.init_device_mesh = lambda *a, **k: None
dm.DeviceMesh = type('DeviceMesh', (), {})
_bind_submodule('torch.distributed', 'device_mesh', dm)

# _composable.fsdp
comp = types.ModuleType('torch.distributed._composable')
comp.__path__ = []
_bind_submodule('torch.distributed', '_composable', comp)
fsdp_mod = types.ModuleType('torch.distributed._composable.fsdp')
fsdp_mod.fully_shard = lambda *a, **k: None
_bind_submodule('torch.distributed._composable', 'fsdp', fsdp_mod)

# _shard / sharding_spec
sh_spec = types.ModuleType('torch.distributed._shard')
sh_spec.__path__ = []
_bind_submodule('torch.distributed', '_shard', sh_spec)
sh_utils = types.ModuleType('torch.distributed._shard.sharding_spec')
sh_utils.ShardingSpec = type('ShardingSpec', (), {})
_bind_submodule('torch.distributed._shard', 'sharding_spec', sh_utils)

# api
api = types.ModuleType('torch.distributed._shard.sharded_tensor')
api.__path__ = []
api.ShardedTensor = type('ShardedTensor', (), {})
_bind_submodule('torch.distributed._shard', 'sharded_tensor', api)
sh_api = types.ModuleType('torch.distributed._shard.sharded_tensor.api')
sh_api.ShardedTensor = type('ShardedTensor', (), {})
_bind_submodule('torch.distributed._shard.sharded_tensor', 'api', sh_api)

# fsdp
fsdp = types.ModuleType('torch.distributed.fsdp')
fsdp.__path__ = []
fsdp.CPUOffloadPolicy = type('CPUOffloadPolicy', (), {})
fsdp.MixedPrecisionPolicy = type('MixedPrecisionPolicy', (), {})
_bind_submodule('torch.distributed', 'fsdp', fsdp)
fsdp_api = types.ModuleType('torch.distributed.fsdp.api')
fsdp_api.CPUOffloadPolicy = type('CPUOffloadPolicy', (), {})
fsdp_api.MixedPrecisionPolicy = type('MixedPrecisionPolicy', (), {})
_bind_submodule('torch.distributed.fsdp', 'api', fsdp_api)
fsdp_utils = types.ModuleType('torch.distributed.fsdp._init_utils')
fsdp_utils._get_fsdp_state = lambda *a, **k: None
_bind_submodule('torch.distributed.fsdp', '_init_utils', fsdp_utils)

# _composable.contract
cont = types.ModuleType('torch.distributed._composable.contract')
cont.contract = lambda *a, **k: (lambda f: f)
_bind_submodule('torch.distributed._composable', 'contract', cont)

# torch.distributed.tensor.experimental._func_map — нужен dynamo
# (should_wrap_in_hop импортирует _local_map_wrapped безусловно, когда
# distributed.is_available()==True); без него ломается torch.compile.
_exp_pkg = types.ModuleType('torch.distributed.tensor.experimental')
_exp_pkg.__path__ = []
_bind_submodule('torch.distributed.tensor', 'experimental', _exp_pkg)
_funcmap = types.ModuleType('torch.distributed.tensor.experimental._func_map')
_funcmap._local_map_wrapped = lambda *a, **k: None
_bind_submodule('torch.distributed.tensor.experimental', '_func_map', _funcmap)

# _functorch/partitioners.py дергает torch.ops._c10d_functional.wait_tensor.default
# при dist.is_available()==True; в этой сборке op не зарегистрирован — подкладываем
# объект-заглушку (сравнение is всегда False, путь распределения выключен).
try:
    class _FakeOp:
        default = object()
    torch.ops._c10d_functional.wait_tensor = _FakeOp()
except Exception:
    pass

try:
    import transformers.models.deepseek_v4.modeling_deepseek_v4 as m
    print('ИМПОРТ УСПЕШЕН:', hasattr(m, 'DeepseekV4ForCausalLM'), hasattr(m, 'DeepseekV4Model'))
except Exception:
    import traceback
    traceback.print_exc()
