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
dist_pkg.Backend = type('Backend', (), {'GLOO': 'gloo', 'NCCL': 'nccl'})
dist_pkg.ProcessGroup = type('ProcessGroup', (), {})
dist_pkg.gather = lambda *a, **k: None
dist_pkg.scatter = lambda *a, **k: None
sys.modules['torch.distributed'] = dist_pkg
torch.distributed = dist_pkg

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
sys.modules['torch.distributed.tensor'] = tensor_pkg

pt = types.ModuleType('torch.distributed.tensor.placement_types')
pt.Replicate = type('Replicate', (), {})
pt.Shard = type('Shard', (), {})
pt.Partial = type('Partial', (), {})
sys.modules['torch.distributed.tensor.placement_types'] = pt

utils = types.ModuleType('torch.distributed.tensor._utils')
utils.compute_local_shape_and_global_offset = lambda *a, **k: (None, None)
sys.modules['torch.distributed.tensor._utils'] = utils

funcol = types.ModuleType('torch.distributed._functional_collectives')
funcol.all_gather_tensor = lambda *a, **k: None
sys.modules['torch.distributed._functional_collectives'] = funcol

c10d = types.ModuleType('torch.distributed.distributed_c10d')
c10d._get_default_group = lambda: None
c10d.get_rank = lambda: 0
c10d.get_world_size = lambda: 1
sys.modules['torch.distributed.distributed_c10d'] = c10d

# device_mesh тоже может понадобиться
dm = types.ModuleType('torch.distributed.device_mesh')
dm.init_device_mesh = lambda *a, **k: None
dm.DeviceMesh = type('DeviceMesh', (), {})
sys.modules['torch.distributed.device_mesh'] = dm

# _composable.fsdp
comp = types.ModuleType('torch.distributed._composable')
comp.__path__ = []
sys.modules['torch.distributed._composable'] = comp
fsdp_mod = types.ModuleType('torch.distributed._composable.fsdp')
fsdp_mod.fully_shard = lambda *a, **k: None
sys.modules['torch.distributed._composable.fsdp'] = fsdp_mod

# _shard / sharding_spec
sh_spec = types.ModuleType('torch.distributed._shard')
sh_spec.__path__ = []
sys.modules['torch.distributed._shard'] = sh_spec
sh_utils = types.ModuleType('torch.distributed._shard.sharding_spec')
sh_utils.ShardingSpec = type('ShardingSpec', (), {})
sys.modules['torch.distributed._shard.sharding_spec'] = sh_utils

# api
api = types.ModuleType('torch.distributed._shard.sharded_tensor')
api.__path__ = []
api.ShardedTensor = type('ShardedTensor', (), {})
sys.modules['torch.distributed._shard.sharded_tensor'] = api
sh_api = types.ModuleType('torch.distributed._shard.sharded_tensor.api')
sh_api.ShardedTensor = type('ShardedTensor', (), {})
sys.modules['torch.distributed._shard.sharded_tensor.api'] = sh_api

# fsdp
fsdp = types.ModuleType('torch.distributed.fsdp')
fsdp.__path__ = []
fsdp.CPUOffloadPolicy = type('CPUOffloadPolicy', (), {})
fsdp.MixedPrecisionPolicy = type('MixedPrecisionPolicy', (), {})
sys.modules['torch.distributed.fsdp'] = fsdp
fsdp_api = types.ModuleType('torch.distributed.fsdp.api')
fsdp_api.CPUOffloadPolicy = type('CPUOffloadPolicy', (), {})
fsdp_api.MixedPrecisionPolicy = type('MixedPrecisionPolicy', (), {})
sys.modules['torch.distributed.fsdp.api'] = fsdp_api
fsdp_utils = types.ModuleType('torch.distributed.fsdp._init_utils')
fsdp_utils._get_fsdp_state = lambda *a, **k: None
sys.modules['torch.distributed.fsdp._init_utils'] = fsdp_utils

# _composable.contract
cont = types.ModuleType('torch.distributed._composable.contract')
cont.contract = lambda *a, **k: (lambda f: f)
sys.modules['torch.distributed._composable.contract'] = cont

try:
    import transformers.models.deepseek_v4.modeling_deepseek_v4 as m
    print('ИМПОРТ УСПЕШЕН:', hasattr(m, 'DeepseekV4ForCausalLM'), hasattr(m, 'DeepseekV4Model'))
except Exception:
    import traceback
    traceback.print_exc()
