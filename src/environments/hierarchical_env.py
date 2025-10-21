import logging
from typing import Any, Dict, List, SupportsFloat, Tuple
import numpy as np
import pandas as pd
from gymnasium import spaces
import glob
import csv
import os

from nsoran.ns_env import NsOranEnv


class HierarchicalEnv(NsOranEnv):
    gnb_state_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "cellId": "INTEGER",
        "state": "INTEGER",
    }

    latency_tracking_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "step": "INTEGER",
        "cell_2_latency": "REAL",
        "cell_3_latency": "REAL",
        "cell_4_latency": "REAL",
        "cell_5_latency": "REAL",
        "cell_6_latency": "REAL",
        "cell_7_latency": "REAL",
        "cell_8_latency": "REAL",
        "avg_cell_latency": "REAL",
        "sum_latency": "REAL",
        "max_latency": "REAL",
        "min_latency": "REAL",
    }

    QOS_CSV_DIR = "/home/eliothluy/ns-o-ran-gym/examples/output"
    QOS_CSV_BASENAME = "qos_all_metrics.csv"

    def __init__(self, ns3_path: str, scenario_configuration: dict, output_folder: str, optimized: bool, verbose: bool = False):
        # usar o mesmo nome para log e control, aceitando lista/tupla do JSON
        raw_control = scenario_configuration.get("controlFileName", "hierarchical_actions.csv")
        if isinstance(raw_control, (list, tuple)):
            raw_control = raw_control[0] if len(raw_control) > 0 else "hierarchical_actions.csv"
        control_basename = os.path.basename(str(raw_control))  # grava em Simulation/

        super().__init__(
            ns3_path=ns3_path,
            scenario="scenario-hierarchical",
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            control_header=["timestamp", "cellId", "hoAllowed, 'ueId','nrCellId"],
            log_file="HieActions.txt",       # MESMO nome
            control_file="hierarchical_actions.csv",   # MESMO nome
        )

        self.log = logging.getLogger(self.__class__.__name__)
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(
                filename="./reward_ts_hie.log",
                level=logging.DEBUG,
                format="%(asctime)s - %(message)s"
            )

        # ---------- Config ----------
        self.cellList = [2, 3, 4, 5, 6, 7, 8]
        self.prefer_action_state = True  # usa os bits ES como estado, se bsState não ajudar
        self.last_es_action = np.ones(len(self.cellList), dtype=int)  # todas ON inicialmente

        # ES: colunas de estado e recompensa
        self.columns_state_es = (
            [f"EEKPI_RL_{c}" for c in self.cellList]
            + [f"ES_ON_COST_{c}" for c in self.cellList]
            + [f"QosFlow.PdcpPduVolumeDL_Filter_{c}" for c in self.cellList]
            + [f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{c}" for c in self.cellList]
            + [f"RLF_Counter_{c}" for c in self.cellList]
            + [f"RLF_VALUE_{c}" for c in self.cellList]
            + [f"RRU_PRBTOTDL_{c}" for c in self.cellList]
            + [f"RRU.PrbUsedDl_{c}" for c in self.cellList]
            + [f"TB_TOTNBRDLINITIAL_64QAM_RATIO_{c}" for c in self.cellList]
            + [
                "SUM_QosFlow.PdcpPduVolumeDL_Filter",
                "SUM_RLF_VALUE",
                "SUM_TB.TotNbrDl.1",
                "SUM_ES_ON_COST",
                "SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)",
                "ZERO_COUNT",
            ]
        )
        self.columns_reward_es = [
            "SUM_QosFlow.PdcpPduVolumeDL_Filter",
            "SUM_TB.TotNbrDl.1",
            "SUM_RLF_VALUE",
            "SUM_ES_ON_COST",
            "SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)",
            "ZERO_COUNT",
        ]

        self.observations = pd.DataFrame()
        self.cells_states: Dict[int, int] = {}
        self.cell_timestamp_state_dict = {cell: float("inf") for cell in self.cellList}
        self.num_steps = 0
        # bits ES "como são" (1=ON, 0=OFF). ZERO_COUNT = nº OFF.
        self.previous_inverted_action = "1111111"

        # TS
        self.columns_state_ts = [
            "RRU.PrbUsedDl",
            "L3 serving SINR",
            "DRB.MeanActiveUeDl",
            "TB.TotNbrDlInitial.Qpsk",
            "TB.TotNbrDlInitial.16Qam",
            "TB.TotNbrDlInitial.64Qam",
            "TB.TotNbrDlInitial",
        ]
        self.columns_reward_ts = ["DRB.UEThpDl.UEID", "nrCellId"]
        self.previous_kpms_ts = None
        self.handovers_dict: Dict[int, int] = {}

        # ES on-cost params
        self.Cf = 1.0
        self.lambdaf = 0.1
        self.time_factor = 0.01

        # ---------- Normalização de recompensa ----------
        self.rew_beta = 0.98  # EMA para escala
        self._abs_reward_ema = 1.0  # escala inicial (evita div/0)

        # tamanhos
        n_gnbs = 7
        ues_cfg = scenario_configuration.get("ues", 0)
        ues_each_gnb = ues_cfg[0] if isinstance(ues_cfg, (list, tuple)) else ues_cfg
        n_ues = int(ues_each_gnb) * n_gnbs

        # spaces
        self.observation_space = spaces.Dict(
            {
                "es_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.columns_state_es),), dtype=np.float32),
                "ts_obs": spaces.Box(shape=(n_ues, len(self.columns_state_ts) + 1), low=-np.inf, high=np.inf, dtype=np.float64),
            }
        )
        n_actions_ue_ts = 9  # 0=no-op, 1..8 = destino (seu runner)
        self.action_space = spaces.Dict({
            "es_action": spaces.MultiBinary(n_gnbs),                   # 1=ON, 0=OFF
            "ts_action": spaces.MultiDiscrete([n_actions_ue_ts] * n_ues)
        })

        # QoS CSV
        self.qos_csv_path = os.path.join(self.QOS_CSV_DIR, self.QOS_CSV_BASENAME)
        self._qos_header = (
            ["timestamp", "step"]
            + self.columns_state_es
            + [c for c in self.columns_reward_es if c not in self.columns_state_es]
            + ["latency_cell_ms", "latency_ue_ms", "reward_raw", "reward_norm"]
        )
        self._ensure_qos_csv()

    # ---------- helper p/ ler configs como escalar (aceita lista ou valor) ----------
    def _cfg(self, key: str, default):
        v = self.scenario_configuration.get(key, default)
        if isinstance(v, (list, tuple)):
            return v[0] if len(v) > 0 else default
        return v

    # ---------- helper para normalizar retornos do datalake ----------
    def _normalize_rows(self, rows, min_len: int | None = None):
        import numpy as _np
        if rows is None:
            return []
        if isinstance(rows, (int, float, _np.integer, _np.floating)):
            return []
        if isinstance(rows, tuple) and (len(rows) == 0 or not isinstance(rows[0], (list, tuple))):
            rows = [rows]
        elif not isinstance(rows, list):
            try:
                rows = list(rows)
            except Exception:
                return []
        norm = []
        for r in rows:
            if isinstance(r, (int, float, _np.integer, _np.floating)):
                r = (float(r),)
            elif not isinstance(r, (list, tuple)):
                continue
            r = tuple(r)
            if min_len is not None and len(r) < min_len:
                continue
            norm.append(r)
        return norm

    # ---------------- hooks do use case ----------------
    def _init_datalake_usecase(self):
        grafana_keys = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "time_grafana": "INTEGER",
            "step": "INTEGER",
            "throughput": "REAL",
            "en_cons": "REAL",
            "rlf": "REAL",
            "on_cost": "REAL",
            "latency_cell_ms": "REAL",
            "latency_ue_ms": "REAL",
            "reward_raw": "REAL",
            "reward_norm": "REAL",
        }
        self.datalake._create_table("bsState", self.gnb_state_keys)
        self.datalake._create_table("grafana", grafana_keys)
        self.datalake._create_table("latency_tracking", self.latency_tracking_keys)
        super()._init_datalake_usecase()

    def _fill_datalake_usecase(self):
        for file_path in glob.glob(os.path.join(self.sim_path, "bsState.txt")):
            try:
                with open(file_path, "r") as csvfile:
                    reader = csv.DictReader(csvfile, delimiter=" ")
                    for row in reader:
                        timestamp = int(row["UNIX"])
                        if timestamp >= self.last_timestamp:
                            db_row = {
                                "timestamp": timestamp,
                                "ueImsiComplete": None,
                                "cellId": int(row["Id"]),
                                "state": int(row["State"]),
                            }
                            self.datalake.insert_data("bsState", db_row)
                            self.last_timestamp = timestamp
            except Exception as e:
                self.log.warning(f"Não foi possível ler {file_path}: {e}")

    # ---------------- Gym API ----------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {"es_obs": self._get_obs_es(), "ts_obs": self._get_obs_ts()}

    def _compute_action(self, action: Dict[str, np.ndarray]) -> List[tuple]:
        """
        1) ES: usa bits como estão (1=ON, 0=OFF)
        2) TS: 0=no-op, >0 aplica valor direto (seu runner mapeia)
        """
        action_list: List[tuple] = []
        ts = int(self.last_timestamp)  # use o timestamp real

        # ---- ES ----
        bits = np.asarray(action["es_action"], dtype=int).flatten()
        # aplicar e registrar bsState (para eventuais leituras)
        for i, cell in enumerate(self.cellList):
            ho_allowed = int(bits[i])  # 1=ON, 0=OFF
            action_list.append(("ES", int(cell), ho_allowed))
            self.datalake.insert_data("bsState", {"timestamp": ts, "ueImsiComplete": None, "cellId": int(cell), "state": ho_allowed})
        # manter para ZERO_COUNT e prefer_action_state
        self.last_es_action = bits.copy()
        self.previous_inverted_action = "".join(str(b) for b in bits.tolist())  # ZERO_COUNT = nº de '0'

        # ---- TS ----
        for ueId, targetCellId in enumerate(action["ts_action"]):
            targetCellId = int(targetCellId)
            if targetCellId > 0:
                action_list.append(("TS", ueId + 1, targetCellId))

        if self.verbose:
            self.log.debug(f"Hierarchical Action list {action_list}")
        return action_list

    def _compute_reward(self) -> float:
        # PPO usará a recompensa ES normalizada
        return self._compute_reward_es()

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[object, SupportsFloat, bool, bool, Dict[str, Any]]:
        if not self.is_simulation_over():
            actions = self._compute_action(action)
            self.action_controller.create_control_action(int(self.last_timestamp), actions)
            self.controlSemaphore.release()
            self._wait_data_availability()
            self._fill_datalake()

        obs = self._get_obs()
        es_reward_norm = self._compute_reward()
        ts_reward = self._compute_reward_ts()

        self.num_steps += 1

        info = {"last_timestamp": self.last_timestamp, "ts_reward": ts_reward, "es_reward": es_reward_norm}
        if self.return_info:
            info.update(self.render())
        return obs, es_reward_norm, self.terminated, self.truncated, info

    # ---------------- ES ----------------
    def _get_obs_es(self) -> np.ndarray:
        try:
            latency_map = self._load_latency_data(self.last_timestamp)
            # leia IMSI + KPIs corretamente
            kpms_raw = [
                "ueImsiComplete",
                "nrCellId",
                "QosFlow.PdcpPduVolumeDL_Filter",
                "TB.TotNbrDl.1",
                "L3 serving SINR",
                "RRU.PrbUsedDl",
                "TB.TotNbrDlInitial.64Qam",
                "TB.TotNbrDlInitial.Qpsk",
                "TB.TotNbrDlInitial.16Qam",
            ]
            ue_kpms = self.datalake.read_kpms(self.last_timestamp, kpms_raw)
            ue_kpms = self._normalize_rows(ue_kpms, min_len=len(kpms_raw))

            self._update_cell_states()

            ue_complete_kpms = []
            for ue_kpm in ue_kpms:
                imsi = int(ue_kpm[0])
                cell_id = int(ue_kpm[1])
                state = int(self.cells_states.get(cell_id, 1))
                lat = latency_map.get(imsi, {"cell_avg": 0.0, "ue_latency": 0.0})
                ue_complete_kpms.append(tuple(ue_kpm) + (lat["cell_avg"], lat["ue_latency"], state))

            columns = (
                ["ueImsiComplete", "nrCellId"]
                + kpms_raw[2:]
                + ["DRB.PdcpSduDelayDl(cellAverageLatency)", "DRB.PdcpSduDelayDl.UEID (pdcpLatency)", "state"]
            )
            df = pd.DataFrame(ue_complete_kpms, columns=columns) if ue_complete_kpms else pd.DataFrame(columns=columns)
            df["timestamp"] = int(self.last_timestamp)
            df, _ = self.getRLFCounter(df, columns)
            df = self.ue_centric_tocell_centric(df)
            self.observations = self.offline_training_preprocessing(df)

            if self.observations.empty:
                return np.zeros(self.observation_space["es_obs"].shape, dtype=np.float32)

            for col in self.columns_state_es:
                if col not in self.observations.columns:
                    self.observations[col] = 0.0

            states = self.observations[self.columns_state_es]
            return np.asarray(states.iloc[0].values, dtype=np.float32)
        except Exception as e:
            self.log.error(f"Falha ao gerar observação ES: {e}")
            return np.zeros(self.observation_space["es_obs"].shape, dtype=np.float32)

    def _compute_reward_es(self) -> float:
        """
        Calcula recompensa ES bruta e retorna a versão normalizada (~[-1,1]).
        """
        try:
            if self.observations.empty:
                return 0.0

            cell_df = self.observations[self.columns_reward_es].copy()
            latency_ms = 0.0
            if "SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)" in self.observations.columns:
                latency_ms = float(self.observations["SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)"].iloc[0])
            latency_normalized = latency_ms / 1000.0

            throughput_term = 0.5 * (1 + float(cell_df["SUM_QosFlow.PdcpPduVolumeDL_Filter"].iloc[0]))
            energy_term = -0.19 * (float(cell_df["SUM_TB.TotNbrDl.1"].iloc[0]) * float(cell_df["ZERO_COUNT"].iloc[0]))
            rlf_term = -0.2 * float(cell_df["SUM_RLF_VALUE"].iloc[0])
            oncost_term = -0.1 * float(cell_df["SUM_ES_ON_COST"].iloc[0])
            latency_term = -0.15 * float(latency_normalized)

            reward_raw = float(throughput_term + energy_term + rlf_term + oncost_term + latency_term)

            # ----- normalização adaptativa -----
            scale = float(self._abs_reward_ema) if self._abs_reward_ema > 1e-6 else 1.0
            reward_norm = float(np.tanh(reward_raw / scale))
            # atualiza EMA depois de usar como escala (evita "lookahead")
            self._abs_reward_ema = float(self.rew_beta * self._abs_reward_ema + (1.0 - self.rew_beta) * abs(reward_raw))

            db_row = {
                "timestamp": int(self.last_timestamp),
                "ueImsiComplete": None,
                "time_grafana": int(self.last_timestamp),
                "step": int(self.num_steps),
                "throughput": float(cell_df["SUM_QosFlow.PdcpPduVolumeDL_Filter"].iloc[0]) * 10 / 10**6,
                "en_cons": float(cell_df["SUM_TB.TotNbrDl.1"].iloc[0]),
                "rlf": float(cell_df["SUM_RLF_VALUE"].iloc[0]),
                "on_cost": float(cell_df["SUM_ES_ON_COST"].iloc[0]),
                "latency_cell_ms": float(latency_ms),
                "latency_ue_ms": float(latency_ms),
                "reward_raw": reward_raw,
                "reward_norm": reward_norm,
            }
            self.datalake.insert_data("grafana", db_row)
            self._store_latency_in_sqlite(latency_ms, latency_ms)
            self._append_qos_snapshot(latency_ms, latency_ms, reward_raw, reward_norm)

            return reward_norm
        except Exception as e:
            self.log.error(f"Falha ao calcular recompensa ES: {e}")
            return 0.0

    # ---------------- TS ----------------
    def _get_obs_ts(self) -> np.ndarray:
        try:
            req = ["ueImsiComplete"] + self.columns_state_ts
            ue_kpms = self.datalake.read_kpms(self.last_timestamp, req)
            ue_kpms = self._normalize_rows(ue_kpms, min_len=len(req))
            if not ue_kpms:
                return np.zeros(self.observation_space["ts_obs"].shape, dtype=np.float64)

            obs_array = np.array(ue_kpms, dtype=np.float64)
            n_ues_obs, n_feats_obs = obs_array.shape
            n_ues_expected, n_feats_expected = self.observation_space["ts_obs"].shape

            if n_ues_obs < n_ues_expected:
                padding = np.zeros((n_ues_expected - n_ues_obs, n_feats_obs))
                obs_array = np.vstack((obs_array, padding))
            elif n_ues_obs > n_ues_expected:
                obs_array = obs_array[:n_ues_expected, :]

            if n_feats_obs < n_feats_expected:
                padding = np.zeros((n_ues_expected, n_feats_expected - n_feats_obs))
                obs_array = np.hstack((obs_array, padding))
            elif n_feats_obs > n_feats_expected:
                obs_array = obs_array[:, :n_feats_expected]

            return obs_array
        except Exception as e:
            self.log.error(f"Falha ao gerar observação TS: {e}")
            return np.zeros(self.observation_space["ts_obs"].shape, dtype=np.float64)

    def _compute_reward_ts(self) -> float:
        total_reward = 0.0
        try:
            req = ["ueImsiComplete"] + self.columns_reward_ts  # (imsi, thpDl, cell)
            current_kpms = self.datalake.read_kpms(self.last_timestamp, req)
            current_kpms = self._normalize_rows(current_kpms, min_len=3)

            if self.previous_kpms_ts is None:
                ip = self._cfg("indicationPeriodicity", 0.1)
                ind_ms = int(ip * 1000)
                prev_ts = self.last_timestamp - ind_ms
                prev_rows = self.datalake.read_kpms(prev_ts, req)
                self.previous_kpms_ts = self._normalize_rows(prev_rows, min_len=3)

            if not current_kpms or not self.previous_kpms_ts:
                if self.verbose:
                    self.log.debug("TS: KPMs atuais/anteriores ausentes ou inválidos; recompensa 0.")
                return 0.0

            prev_map = {}
            for row in self.previous_kpms_ts:
                try:
                    imsi_o, thp_o, cell_o = row[:3]
                    prev_map[int(imsi_o)] = (float(thp_o), int(cell_o))
                except Exception:
                    continue

            for t_n in current_kpms:
                try:
                    ueImsi_n, ueThpDl_n, currentCell = t_n[:3]
                    ueImsi_n = int(ueImsi_n)
                    ueThpDl_n = float(ueThpDl_n)
                    currentCell = int(currentCell)
                except Exception:
                    continue

                if ueImsi_n not in prev_map:
                    continue

                ueThpDl_o, sourceCell = prev_map[ueImsi_n]
                HoCost = 0.0
                if currentCell != sourceCell:
                    lastHo = self.handovers_dict.get(ueImsi_n, 0)
                    if lastHo != 0:
                        timeDiff = (self.last_timestamp - lastHo) * self.time_factor
                        HoCost = self.Cf * ((1 - self.lambdaf) ** timeDiff)
                    self.handovers_dict[ueImsi_n] = self.last_timestamp

                LogOld = np.log10(ueThpDl_o) if ueThpDl_o > 0 else 0.0
                LogNew = np.log10(ueThpDl_n) if ueThpDl_n > 0 else 0.0
                total_reward += float((LogNew - LogOld) - HoCost)

            self.previous_kpms_ts = current_kpms
            return float(total_reward)

        except Exception as e:
            self.log.error(f"Falha ao calcular recompensa TS: {e}")
            return 0.0

    # ---------------- utilidades ES ----------------
    def _ensure_qos_csv(self):
        os.makedirs(os.path.dirname(self.qos_csv_path) or ".", exist_ok=True)
        new_file = (not os.path.exists(self.qos_csv_path)) or os.path.getsize(self.qos_csv_path) == 0
        if new_file:
            with open(self.qos_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._qos_header)

    def _append_qos_snapshot(self, latency_cell_ms: float, latency_ue_ms: float, reward_raw: float, reward_norm: float):
        row: List[float] = []
        row.append(int(self.last_timestamp))
        row.append(int(self.num_steps))

        for col in self.columns_state_es:
            val = 0.0
            try:
                if col in self.observations.columns:
                    val = float(self.observations[col].iloc[0])
            except Exception:
                pass
            row.append(val)

        for col in self.columns_reward_es:
            if col in self.columns_state_es:
                continue
            val = 0.0
            try:
                if col in self.observations.columns:
                    val = float(self.observations[col].iloc[0])
            except Exception:
                pass
            row.append(val)

        row.append(float(latency_cell_ms))
        row.append(float(latency_ue_ms))
        row.append(float(reward_raw))
        row.append(float(reward_norm))

        with open(self.qos_csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _load_latency_data(self, timestamp: int) -> Dict[int, Dict[str, float]]:
        try:
            latency_table_data = self.datalake.read_table("lte_cu_up") or []
            if not latency_table_data:
                return {}

            latency_rows = [row for row in latency_table_data if int(row[0]) == int(timestamp)]
            if not latency_rows:
                return {}

            latency_map: Dict[int, Dict[str, float]] = {}
            for row in latency_rows:
                try:
                    imsi = int(row[1])
                    cell_avg = float(row[3]) if len(row) > 3 and row[3] not in (None, "", 0) else 0.0
                    ue_latency = float(row[8]) if len(row) > 8 and row[8] not in (None, "", 0) else 0.0
                    final_latency = ue_latency if ue_latency > 0 else cell_avg
                    latency_map[imsi] = {"cell_avg": cell_avg, "ue_latency": final_latency}
                except (ValueError, IndexError, TypeError):
                    continue
            return latency_map
        except Exception as e:
            self.log.error("Falha ao carregar latência: %r", e)
            return {}

    def _store_latency_in_sqlite(self, avg_latency: float, sum_latency: float):
        try:
            latency_dict: Dict[str, float] = {"timestamp": int(self.last_timestamp), "ueImsiComplete": None, "step": int(self.num_steps)}
            per_cell_latencies: List[float] = []
            for cell in self.cellList:
                col_name = f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}"
                lat_val = float(self.observations[col_name].iloc[0]) if col_name in self.observations.columns else 0.0
                latency_dict[f"cell_{cell}_latency"] = lat_val
                per_cell_latencies.append(lat_val)
            latency_dict["avg_cell_latency"] = float(avg_latency)
            latency_dict["sum_latency"] = float(sum_latency)
            non_zero = [lat for lat in per_cell_latencies if lat > 0]
            latency_dict["max_latency"] = float(max(non_zero)) if non_zero else 0.0
            latency_dict["min_latency"] = float(min(non_zero)) if non_zero else 0.0
            self.datalake.insert_data("latency_tracking", latency_dict)
        except Exception as e:
            self.log.error("Falha ao armazenar latência: %r", e)

    def ue_centric_tocell_centric(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ueImsiComplete" in df.columns:
            df.drop(columns=["ueImsiComplete"], inplace=True)
        if "L3 serving SINR" in df.columns:
            df["L3 serving SINR"] = df["L3 serving SINR"].replace(-np.inf, 0)
            df.drop(columns=["L3 serving SINR"], inplace=True)
        df = df.drop_duplicates()
        df.reset_index(drop=True, inplace=True)
        return df

    def rename_columns(self, columns: List[str], cell_no: int) -> List[str]:
        return [f"{c}_{cell_no}" for c in columns]

    def offline_training_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_eekpi_qpsk_16_64qam_sum_and_ratio(df)
        df.sort_values(by=["timestamp"], ascending=True, inplace=True)

        if "state" in df.columns:
            # invertido para "custo de ligar": 1 se OFF, 0 se ON
            df["state"] = df["state"].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))

        cell_df = pd.DataFrame()
        is_initial_cell = True

        for cell in self.cellList:
            temp = df.loc[df["nrCellId"] == cell].copy()
            if "RRU.PrbUsedDl" in temp.columns:
                temp["RRU_PRBTOTDL"] = (temp["RRU.PrbUsedDl"] / 139) * 100
            else:
                temp["RRU_PRBTOTDL"] = 0.0

            tb = temp.get("TB.TotNbrDl.1", pd.Series([1e-5] * len(temp)))
            qos = temp.get("QosFlow.PdcpPduVolumeDL_Filter", pd.Series([0.0] * len(temp)))
            temp["EEKPI_RL"] = qos / tb.replace(0, 1e-5)

            temp.columns = self.rename_columns(list(temp.columns), cell)
            temp.rename(columns={f"timestamp_{cell}": "timestamp"}, inplace=True)

            alias_cols = [
                f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}",
                f"DRB.PdcpSduDelayDl(cellAverageLatency)_{cell}",
            ]
            found = None
            for c in alias_cols:
                if c in temp.columns:
                    found = c
                    break
            if found is not None:
                temp.rename(columns={found: f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}"}, inplace=True)
            else:
                temp[f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}"] = 0.0

            if is_initial_cell:
                cell_df = temp
                is_initial_cell = False
            else:
                cell_df = pd.merge(cell_df, temp, how="outer", on=["timestamp"])

        cell_df = cell_df.infer_objects(copy=False).fillna(0)
        cell_df = self.es_on_cost_calculation(cell_df)

        agg_specs = {
            "SUM_QosFlow.PdcpPduVolumeDL_Filter": ("QosFlow.PdcpPduVolumeDL_Filter_", "sum"),
            "SUM_TB.TotNbrDl.1": ("TB.TotNbrDl.1_", "sum"),
            "SUM_ES_ON_COST": ("ES_ON_COST_", "sum"),
            "SUM_RLF_VALUE": ("RLF_VALUE_", "sum"),
            "SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)": ("DRB.PdcpSduDelayDl.UEID (pdcpLatency)_", "mean"),
        }
        for out_col, (prefix, how) in agg_specs.items():
            cols = cell_df.filter(like=prefix)
            if cols.shape[1] == 0:
                cell_df[out_col] = 0.0
            elif how == "sum":
                cell_df[out_col] = cols.sum(axis=1)
            elif how == "mean":
                cell_df[out_col] = cols.mean(axis=1)
            else:
                cell_df[out_col] = 0.0

        # ZERO_COUNT: conta quantos OFF (0) nos bits ES atuais
        cell_df["ACTION_BINARY"] = self.previous_inverted_action  # agora é a ação ES direta
        cell_df["ACTION_BINARY"] = cell_df["ACTION_BINARY"].astype(str)
        cell_df["ZERO_COUNT"] = cell_df["ACTION_BINARY"].apply(lambda x: x.count("0"))

        return cell_df

    def add_eekpi_qpsk_16_64qam_sum_and_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        if {"TB.TotNbrDlInitial.Qpsk", "TB.TotNbrDlInitial.16Qam", "TB.TotNbrDlInitial.64Qam"}.issubset(df.columns):
            df["TB.TOTNBRDLINITIAL.SUM"] = df["TB.TotNbrDlInitial.Qpsk"] + df["TB.TotNbrDlInitial.16Qam"] + df["TB.TotNbrDlInitial.64Qam"]
            df["TB_TOTNBRDLINITIAL_64QAM_RATIO"] = (df["TB.TotNbrDlInitial.64Qam"] / df["TB.TOTNBRDLINITIAL.SUM"]).replace([np.inf, -np.inf], 0).fillna(0.00001)
        if "RRU.PrbUsedDl" in df.columns:
            df["RRU.PrbUsedDl"] = df["RRU.PrbUsedDl"].replace(0, 0.00001)
        if "TB.TotNbrDl.1" in df.columns:
            df["TB.TotNbrDl.1"] = df["TB.TotNbrDl.1"].replace(0, 0.00001)
        return df

    def getRLFCounter(self, df: pd.DataFrame, columns: List[str]):
        df["timestamp"] = df["timestamp"].astype(int)
        df["RLF_Counter"] = 0.0
        df["RLF_VALUE"] = 0
        if "L3 serving SINR" in df.columns:
            df["L3 serving SINR"] = df["L3 serving SINR"].replace(-np.inf, 0)
        if "nrCellId" not in df.columns or "L3 serving SINR" not in df.columns:
            return df, columns

        grouped = df.groupby(["timestamp", "nrCellId"])
        for (timestamp, cell), group in grouped:
            total_count = group.shape[0]
            if total_count > 0:
                num_values = group[group["L3 serving SINR"] < -5].shape[0]
                rlf_value = (num_values / total_count) * 100
                mask = (df["timestamp"] == timestamp) & (df["nrCellId"] == cell)
                df.loc[mask, "RLF_Counter"] = rlf_value
                df.loc[mask, "RLF_VALUE"] = num_values
        return df, columns

    def es_on_cost_calculation(self, cell_df: pd.DataFrame) -> pd.DataFrame:
        """
        Custo aplicado quando a célula está OFF (state=0).
        """
        for cell in self.cellList:
            current_timestamp = int(self.last_timestamp)
            time_diff_obs: List[float] = []

            # estado atual: 1=ON, 0=OFF
            current_state = int(self.cells_states.get(cell, 1))

            if current_state == 0:  # OFF
                if self.cell_timestamp_state_dict[cell] == float("inf"):
                    # acabou de ficar OFF -> custo de ligar na próxima vez
                    time_diff_obs.append(100)
                    self.cell_timestamp_state_dict[cell] = current_timestamp
                else:
                    # permanece OFF -> acumula tempo OFF
                    time_diff_obs.append(current_timestamp - self.cell_timestamp_state_dict[cell] + 100)
            else:  # ON
                time_diff_obs.append(float("inf"))
                self.cell_timestamp_state_dict[cell] = float("inf")

            time_diff_obs_col = f"TIME_DIFF_OBS_{cell}"
            cell_df[time_diff_obs_col] = time_diff_obs * max(1, len(cell_df))

            es_on_cost_col = f"ES_ON_COST_{cell}"
            cell_df[es_on_cost_col] = cell_df[time_diff_obs_col].apply(
                lambda diff: self.Cf * ((1 - self.lambdaf) ** (diff * self.time_factor)) if diff != float("inf") else 0
            )

        return cell_df

    def _update_cell_states(self):
        """
        Atualiza cache de estados das células.
        Se prefer_action_state=True, usa diretamente os bits da última ação ES.
        Caso contrário, tenta ler o snapshot do datalake (bsState).
        """
        if getattr(self, "prefer_action_state", False) and hasattr(self, "last_es_action") and self.last_es_action is not None:
            self.cells_states = {cell: int(self.last_es_action[i]) for i, cell in enumerate(self.cellList)}
            return

        cell_states_table = self.datalake.read_table("bsState") or []
        self.cells_states = {cell: 1 for cell in self.cellList}  # default ON

        if not cell_states_table:
            return

        # alvo: snapshot imediatamente anterior (ou igual) ao timestamp atual
        ip = float(self._cfg("indicationPeriodicity", 0.1))
        step_ms = max(int(ip * 1000.0), 1)
        target_ts = int(self.last_timestamp) - step_ms

        candidates = [row for row in cell_states_table if int(row[0]) <= target_ts]
        if candidates:
            selected_ts = max(int(r[0]) for r in candidates)
        else:
            candidates2 = [row for row in cell_states_table if int(row[0]) <= int(self.last_timestamp)]
            if candidates2:
                selected_ts = max(int(r[0]) for r in candidates2)
            else:
                selected_ts = max(int(r[0]) for r in cell_states_table)

        states_at_ts = [r for r in cell_states_table if int(r[0]) == selected_ts]
        by_cell: Dict[int, int] = {}
        for ts, _, cellId, state in states_at_ts:
            by_cell[int(cellId)] = int(state)

        for cellId in self.cellList:
            self.cells_states[cellId] = int(by_cell.get(cellId, 1))
