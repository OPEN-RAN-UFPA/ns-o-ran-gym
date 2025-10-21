# hierarchical_control.py
import argparse
import json
import logging
import time
from environments.hierarchical_env import HierarchicalEnv
# (Reutilizando utils de log do es_env, se disponíveis)
try:
    from energy_saving import _print_qos_tuple, _grab_qos_from_grafana
except ImportError:
    print("Aviso: 'energy_saving.py' não encontrado. Funções de log de QoS podem falhar.")
    _print_qos_tuple = lambda x: print(f"QoS Tuple: {x}")
    _grab_qos_from_grafana = lambda x: None

# Configura o logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("HRL_Runner")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the hierarchical (TS+ES) environment")
    
    # Argumentos (padrões baseados nos seus outros scripts)
    parser.add_argument("--config", type=str, default="/home/eliothluy/ns-o-ran-gym/src/environments/scenario_configurations/hierarchical_use_case.json",
                        help="Path to the configuration file (usa ES como base)")
    parser.add_argument("--output_folder", type=str, default="/home/eliothluy/ns-o-ran-gym/examples/output",
                        help="Path to the output folder")
    parser.add_argument("--ns3_path", type=str, default="/home/eliothluy/ns-3-mmwave-oran/",
                        help="Path to the ns-3 mmWave O-RAN environment")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of steps to run in the environment")
    parser.add_argument("--optimized", action="store_true",
                        help="Enable optimization mode")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    try:
        with open(args.config) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        log.error(f"Cannot open '{args.config}' file, exiting")
        exit(-1)

    scenario_configuration = json.loads(params)
    
    # **IMPORTANTE**: Garante que a heurística de ES esteja desligada
    # para que o agente de RL (ou ações aleatórias) tenha controle total.
    scenario_configuration['heuristicType'] = [-1] 
    
    # **IMPORTANTE**: Define o nome do arquivo de controle unificado
    # que o scenario-hierarchical.cc espera.
    scenario_configuration['controlFileName'] = ["hierarchical_actions.csv"]

    log.info('Creating Hierarchical Environment')
    env = HierarchicalEnv(
        ns3_path=args.ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=args.output_folder,
        optimized=args.optimized,
        verbose=args.verbose
    )
    log.info('Environment Created!')

    log.info('Launch reset ...')
    obs, info = env.reset()
    log.info('... reset done')

    log.info(f'First ES observation shape: {obs["es_obs"].shape}')
    log.info(f'First TS observation shape: {obs["ts_obs"].shape}')
    log.info(f'First Info dict: {info}')

    for step in range(2, args.num_steps):
        # Gera uma ação aleatória para ambos os níveis
        action = env.action_space.sample() 
        
        log.info(f'--- Step {step} ---')
        
        # O agente HRL real dividiria 'action' e aplicaria
        # a ação 'es_action' apenas a cada N passos.
        # Aqui, estamos enviando ambas em todos os passos para teste.
        
        obs, reward, terminated, truncated, info = env.step(action)

        log.info(f'ES Obs shape: {obs["es_obs"].shape}')
        log.info(f'TS Obs shape: {obs["ts_obs"].shape}')
        log.info(f'System (ES) Reward: {reward:.4f}')
        log.info(f'TS Reward (from info): {info.get("ts_reward", "N/A"):.4f}')
        log.info(f'Terminated: {terminated}, Truncated: {truncated}')

        # Tenta imprimir o log de QoS formatado
        try:
            qos_rec = _grab_qos_from_grafana(env)
            if qos_rec:
                _print_qos_tuple(qos_rec)
        except Exception:
            pass # Falha silenciosa se o helper não estiver disponível

        if terminated or truncated:
            log.warning("Simulation ended.")
            break
            
        time.sleep(0.1) # Pequena pausa para legibilidade

    log.info("Closing environment.")
    env.close()