# environments/__init__.py
import gymnasium as gym
from gymnasium.envs.registration import register, registry

def _safe_register(id: str, **kwargs):
    # Se já existe, não registra de novo
    try:
        gym.spec(id)
        return
    except gym.error.Error:
        pass

    # Evita conflito: se tentar registrar sem versão e já existir alguma -vX, não registra
    if "-v" not in id:
        prefix = id
        if any(spec.id.startswith(prefix + "-v") for spec in registry.values()):
            # Já existe uma versão; pular o alias sem versão para evitar RegistrationError
            return

    register(id=id, **kwargs)

# ---------- Traffic Steering ----------
_safe_register(
    id="TrafficSteeringEnv-v0",
    entry_point="environments.ts_env:TrafficSteeringEnv",
    max_episode_steps=100,
)

# ---------- Energy Saving ----------
_safe_register(
    id="EnergySavingEnv-v0",
    entry_point="environments.es_env:EnergySavingEnv",
    max_episode_steps=100,
)

# ---------- Hierarchical ----------
_safe_register(
    id="HierarchicalEnv-v0",
    entry_point="environments.hierarchical_env:HierarchicalEnv",
    max_episode_steps=500,  # ajuste se quiser
)
