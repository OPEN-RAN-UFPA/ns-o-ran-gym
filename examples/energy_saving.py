#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import csv
from typing import Optional, List, Any

# OBS: ajuste o import conforme a sua árvore de pastas.
# Se "environments" não estiver no PYTHONPATH, rode via: python -m examples.energy_saving
from environments.es_env import EnergySavingEnv

# =========================
# Utils
# =========================
def tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_csv_header(csv_path: str, header: List[str]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    if new_file:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def _append_qos_row(csv_path: str, row: List[Any]) -> None:
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _grab_qos_from_grafana(env: EnergySavingEnv):
    """
    Lê a última linha de métricas da tabela 'grafana' para o timestamp atual.
    Suporta 2 esquemas:
      A) sem latência: [ts, _, time_graf, step, thr, en_cons, rlf, on_cost, reward]
      B) com latência: [ts, _, time_graf, step, thr, en_cons, rlf, on_cost, lat_cell, lat_ue, reward]
    Retorna:
      (timestamp, step, throughput, en_cons, rlf, on_cost, latency_cell_ms, latency_ue_ms, reward)
    """
    try:
        rows = env.datalake.read_table("grafana")  # lista de tuplas
    except Exception:
        return None
    if not rows:
        return None

    # pega a última linha do timestamp atual, se existir; senão, a última disponível
    ts = env.last_timestamp
    candidates = [r for r in rows if len(r) >= 9 and int(r[0]) == int(ts)]
    row = candidates[-1] if candidates else rows[-1]

    # Esquema com latências (>= 11 colunas)
    if len(row) >= 11:
        try:
            return (
                int(row[0]),          # timestamp
                int(row[3]),          # step
                float(row[4]),        # throughput
                float(row[5]),        # en_cons
                float(row[6]),        # rlf
                float(row[7]),        # on_cost
                float(row[8]),        # latency_cell_ms
                float(row[9]),        # latency_ue_ms
                float(row[10]),       # reward
            )
        except Exception:
            pass

    # Esquema sem latências (>= 9 colunas)
    if len(row) >= 9:
        try:
            return (
                int(row[0]),
                int(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
                "",                    # latency_cell_ms ausente
                "",                    # latency_ue_ms ausente
                float(row[8]),
            )
        except Exception:
            pass

    return None


def _print_qos_tuple(rec: tuple) -> None:
    ts, step, thr, en_cons, rlf, on_cost, lat_cell, lat_ue, reward = rec
    thr_str = f"{thr:.3f}" if isinstance(thr, (int, float)) else str(thr)
    latc_str = f"{lat_cell:.3f}" if isinstance(lat_cell, (int, float)) else "-"
    latu_str = f"{lat_ue:.3f}" if isinstance(lat_ue, (int, float)) else "-"
    print(
        f"QoS | ts={ts} step={step} | thr={thr_str} Mbps | en_cons={en_cons:.3f} | "
        f"rlf={rlf:.3f} | on_cost={on_cost:.3f} | lat_cell={latc_str} ms | lat_ue={latu_str} ms | "
        f"reward={reward:.4f}"
    )


# =========================
# Modo heurístico (original)
# =========================
def run_heuristic(env: EnergySavingEnv, num_steps: int, qos_csv: Optional[str]):
    if qos_csv:
        header = ["timestamp", "step", "throughput", "en_cons", "rlf", "on_cost", "latency_cell_ms", "latency_ue_ms", "reward"]
        _ensure_csv_header(qos_csv, header)

    print("Launch reset ", end="", flush=True)
    obs, info = env.reset()
    print("done")
    print(f"First set of observations {obs}")
    print(f"Info {info}")

    # tenta imprimir QoS do primeiro instante (se já houver)
    first_rec = _grab_qos_from_grafana(env)
    if first_rec is not None:
        _print_qos_tuple(first_rec)
        if qos_csv:
            _append_qos_row(qos_csv, list(first_rec))

    for step in range(2, num_steps):
        # lê último estado das células no datalake
        cell_states_table = env.datalake.read_table("bsState")
        states_of_interest = [row for row in cell_states_table if row[0] == env.last_timestamp]
        model_action = [state[3] for state in states_of_interest]

        print(f"Step {step} ", end="", flush=True)
        obs, reward, terminated, truncated, info = env.step(model_action)
        print("done", flush=True)

        print(f"Status t = {step}")
        print(f"Actions {env._compute_action(model_action)}")
        print(f"Observations {obs}")
        print(f"Reward {reward}")
        print(f"Terminated {terminated}")
        print(f"Truncated {truncated}")
        print(f"Info {info}")

        # extrai e imprime QoS (e salva no CSV se habilitado)
        rec = _grab_qos_from_grafana(env)
        if rec is not None:
            _print_qos_tuple(rec)
            if qos_csv:
                _append_qos_row(qos_csv, list(rec))

        if terminated:
            break
        if truncated:
            break  # no treino resetaria; aqui encerramos como no original


# =========================
# PPO (treino) com callback
# =========================
class QoSCSVCallback:  # callback simples sem herdar SB3 BaseCallback (fallback)
    def __init__(self, csv_path: str | None):
        self.csv_path = csv_path
        if self.csv_path:
            _ensure_csv_header(
                self.csv_path,
                ["timestamp", "step", "throughput", "en_cons", "rlf", "on_cost", "latency_cell_ms", "latency_ue_ms", "reward"],
            )

    def on_step(self, vec_env) -> None:
        try:
            env0 = vec_env.envs[0]  # Monitor -> EnergySavingEnv
            base_env = getattr(env0, "env", env0)
            rec = _grab_qos_from_grafana(base_env)
            if rec is not None:
                # imprime QoS periodicamente (a cada step do rollout)
                _print_qos_tuple(rec)
                if self.csv_path:
                    _append_qos_row(self.csv_path, list(rec))
        except Exception:
            pass


def run_ppo(ns3_path: str, scenario_configuration: dict, output_folder: str, optimized: bool,
            total_timesteps: int, seed: int, tb: bool, qos_csv: Optional[str]):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as e:
        print("Stable-Baselines3 não encontrado. Instale com: pip install stable-baselines3", file=sys.stderr)
        raise

    print("Criando ES Environment (para treino)")
    def make_env():
        env = EnergySavingEnv(
            ns3_path=ns3_path,
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    tb_dir = os.path.join(output_folder, "tb") if tb and tensorboard_available() else None
    if tb and tb_dir is None:
        print("Aviso: TensorBoard não disponível — desativando logs TB.")

    print(f"Iniciando PPO | timesteps={total_timesteps} | seed={seed}")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        verbose=1,
        tensorboard_log=tb_dir,
    )

    # callback para imprimir e (opcionalmente) gravar QoS durante o treino
    qos_cb = QoSCSVCallback(qos_csv) if qos_csv is not None else QoSCSVCallback(None)
    try:
        from stable_baselines3.common.callbacks import BaseCallback
        class _SB3QoSCallback(BaseCallback):
            def __init__(self, qc: QoSCSVCallback, verbose: int = 0):
                super().__init__(verbose)
                self.qc = qc
            def _on_step(self) -> bool:
                self.qc.on_step(self.model.get_env())
                return True
        cb_list = [_SB3QoSCallback(qos_cb)]
    except Exception:
        cb_list = None

    model.learn(total_timesteps=total_timesteps, callback=cb_list)

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, "ppo_energy_saving.zip")
    model.save(save_path)
    print(f"Modelo salvo em: {save_path}")

    # avaliação rápida (1 episódio) — imprime e também grava QoS
    print("Avaliação rápida (1 episódio) ...")
    env_eval = make_env()
    obs, info = env_eval.reset()
    ep_return, ep_len = 0.0, 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_eval.step(action)
        ep_return += float(reward)
        ep_len += 1
        rec = _grab_qos_from_grafana(env_eval.env if hasattr(env_eval, "env") else env_eval)
        if rec is not None:
            _print_qos_tuple(rec)
            if qos_csv:
                _append_qos_row(qos_csv, list(rec))
        if terminated or truncated:
            break
    print(f"Eval retorno={ep_return:.3f} | len={ep_len}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the energy saving environment (heuristic or PPO).")
    parser.add_argument("--config", type=str, default="/home/eliothluy/ns-o-ran-gym/src/environments/scenario_configurations/es_use_case.json",
                        help="Path to the configuration file")
    parser.add_argument("--output_folder", type=str, default="/home/eliothluy/ns-o-ran-gym/examples/output",
                        help="Path to the output folder")
    parser.add_argument("--ns3_path", type=str, default="/home/eliothluy/ns-3-mmwave-oran/",
                        help="Path to the ns-3 mmWave O-RAN environment")

    parser.add_argument("--mode", choices=["heuristic", "train"], default="train",
                        help="Execution mode: heuristic loop (original) or PPO training")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps to run (heuristic mode)")
    parser.add_argument("--optimized", action="store_true",
                        help="Enable optimization mode")

    # PPO args
    parser.add_argument("--total_timesteps", type=int, default=20000, help="PPO total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard if available")

    # QoS CSV (default para a pasta pedida)
    parser.add_argument("--qos_csv", type=str, default="/home/eliothluy/ns-o-ran-gym/examples/output/qos_metrics_energy_saving.csv",
                        help="Caminho do CSV para salvar métricas QoS por passo (use '' para desativar)")

    args = parser.parse_args()

    # carrega config
    try:
        with open(args.config) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{args.config}' file, exiting")
        sys.exit(-1)

    scenario_configuration = json.loads(params)
    qos_csv = args.qos_csv if args.qos_csv and args.qos_csv.strip() else None

    if args.mode == "heuristic":
        print("Creating ES Environment")
        env = EnergySavingEnv(
            ns3_path=args.ns3_path,
            scenario_configuration=scenario_configuration,
            output_folder=args.output_folder,
            optimized=args.optimized
        )
        print("Environment Created!")
        run_heuristic(env, args.num_steps, qos_csv)
    else:
        run_ppo(
            ns3_path=args.ns3_path,
            scenario_configuration=scenario_configuration,
            output_folder=args.output_folder,
            optimized=args.optimized,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            tb=args.tensorboard,
            qos_csv=qos_csv
        )
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
