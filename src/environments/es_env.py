from typing_extensions import override

import numpy as np
import pandas as pd
import logging
from nsoran.ns_env import NsOranEnv
from gymnasium import spaces
import glob
import csv
import os


class EnergySavingEnv(NsOranEnv):
    """
    Energy Saving Environment with full latency integration.
    
    Features:
    - Cells 2..8
    - Latency capture from lte_cu_up table (separate from du table)
    - Multi-objective reward: throughput, energy, RLF, on-cost, and LATENCY
    - Per-cell and aggregated latency tracking in SQLite
    - QoS metrics CSV export with latency
    - FIXED: Correct column indices for latency (row[3] not row[2])
    """

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

    def __init__(
        self,
        ns3_path: str,
        scenario_configuration: dict,
        output_folder: str,
        optimized: bool,
        do_heuristic: bool = True,
    ):
        super().__init__(
            ns3_path=ns3_path,
            scenario="scenario-three",
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            control_header=["timestamp", "cellId", "hoAllowed"],
            log_file="EsActions.txt",
            control_file="es_actions_for_ns3.csv",
        )

        self.log = logging.getLogger(self.__class__.__name__)
        self.folder_name = "Simulation"
        self.ns3_simulation_time = scenario_configuration["simTime"] * 1000
        self.cellList = [2, 3, 4, 5, 6, 7, 8]

        self.columns_state = (
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

        self.columns_reward = [
            "SUM_QosFlow.PdcpPduVolumeDL_Filter",
            "SUM_TB.TotNbrDl.1",
            "SUM_RLF_VALUE",
            "SUM_ES_ON_COST",
            "SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)",
            "ZERO_COUNT",
        ]

        self.action_list = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22,
            24, 25, 26, 28, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 48, 49, 50, 52, 56,
            64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 76, 80, 81, 82, 84, 88, 96, 97, 98, 100,
            104, 112,
        ]

        self.observations = []
        self.cells_states = {}
        self.cell_timestamp_state_dict = {cell: float("inf") for cell in self.cellList}
        self.Cf = 1
        self.lambdaf = 0.1
        self.time_factor = 0.01
        self.heur = do_heuristic
        self.num_steps = 0
        self.previous_inverted_action = "0000000"

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.columns_state),), dtype=np.float32
        )

        self.action_space = spaces.MultiBinary(len(self.cellList))

        self.qos_csv_path = os.path.join(self.QOS_CSV_DIR, self.QOS_CSV_BASENAME)
        self._qos_header = (
            ["timestamp", "step"]
            + self.columns_state
            + [c for c in self.columns_reward if c not in self.columns_state]
            + ["latency_cell_ms", "latency_ue_ms", "reward"]
        )

        self._ensure_qos_csv()

    def _ensure_qos_csv(self):
        os.makedirs(os.path.dirname(self.qos_csv_path) or ".", exist_ok=True)
        new_file = (not os.path.exists(self.qos_csv_path)) or os.path.getsize(self.qos_csv_path) == 0
        if new_file:
            with open(self.qos_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self._qos_header)

    def _append_qos_snapshot(self, latency_cell_ms: float, latency_ue_ms: float, reward: float):
        row = []
        row.append(int(self.last_timestamp))
        row.append(int(self.num_steps))

        for col in self.columns_state:
            val = 0.0
            try:
                if col in self.observations.columns:
                    val = float(self.observations[col].iloc[0])
            except Exception:
                pass
            row.append(val)

        for col in self.columns_reward:
            if col in self.columns_state:
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
        row.append(float(reward))

        with open(self.qos_csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _load_latency_data(self, timestamp):
        """
        Load latency data from lte_cu_up table with CORRECT column indices.
        
        Schema from datalake.py lte_cu_up_keys:
        [0] timestamp
        [1] ueImsiComplete
        [2] cellId (INTEGER - was causing 1.000 bug!)
        [3] DRB.PdcpSduDelayDl(cellAverageLatency) ← CORRECT INDEX
        [4] m_pDCPBytesDL(cellDlTxVolume)
        [5] DRB.PdcpSduVolumeDl_Filter.UEID (txBytes)
        [6] Tot.PdcpSduNbrDl.UEID (txDlPackets)
        [7] DRB.PdcpSduBitRateDl.UEID (pdcpThroughput)
        [8] DRB.PdcpSduDelayDl.UEID (pdcpLatency) ← CORRECT INDEX
        """
        try:
            latency_table_data = self.datalake.read_table("lte_cu_up") or []
            
            if not latency_table_data:
                self.log.warning("lte_cu_up table is EMPTY!")
                return {}
            
            latency_rows = [row for row in latency_table_data if int(row[0]) == int(timestamp)]
            
            if not latency_rows:
                self.log.debug("No latency data for timestamp %s", timestamp)
                return {}
            
            latency_map = {}
            for row in latency_rows:
                try:
                    imsi = int(row[1])  # ueImsiComplete
                    
                    # CRITICAL FIX: Use row[3] for cell average, NOT row[2] (which is cellId=1)
                    cell_avg = float(row[3]) if len(row) > 3 and row[3] not in (None, '', 0) else 0.0
                    ue_latency = float(row[8]) if len(row) > 8 and row[8] not in (None, '', 0) else 0.0
                    
                    # Use UE-specific if available, else cell average
                    final_latency = ue_latency if ue_latency > 0 else cell_avg
                    
                    latency_map[imsi] = {
                        'cell_avg': cell_avg,
                        'ue_latency': final_latency
                    }
                    
                    # DEBUG: Log first few UEs to verify correct values
                    if len(latency_map) <= 3:
                        self.log.info(
                            "UE %05d: cell_avg=%.3f ms | ue_latency=%.3f ms | final=%.3f ms",
                            imsi, cell_avg, ue_latency, final_latency
                        )
                        
                except (ValueError, IndexError, TypeError) as e:
                    self.log.warning("Error parsing latency for IMSI %s: %r", row[1] if len(row) > 1 else '?', e)
                    continue
            
            self.log.info(
                "Loaded latency for %d UEs | ts=%s",
                len(latency_map), timestamp
            )
            
            return latency_map
            
        except Exception as e:
            self.log.error("Failed to load latency from lte_cu_up: %r", e)
            return {}

    @override
    def _get_obs(self):
        """
        Get observations with latency loaded from lte_cu_up table.
        """
        # Load latency data from lte_cu_up table (separate from du table)
        latency_map = self._load_latency_data(self.last_timestamp)
        
        # Query regular KPMs from du table
        kpms_raw = [
            "nrCellId",
            "QosFlow.PdcpPduVolumeDL_Filter",
            "TB.TotNbrDl.1",
            "L3 serving SINR",
            "RRU.PrbUsedDl",
            "TB.TotNbrDlInitial.64Qam",
            "TB.TotNbrDlInitial.Qpsk",
            "TB.TotNbrDlInitial.16Qam",
        ]
        
        ue_kpms = self.datalake.read_kpms(self.last_timestamp, kpms_raw) or []
        
        self._update_cell_states()
        
        # Merge KPMs with latency data
        ue_complete_kpms = []
        for ue_kpm in ue_kpms:
            imsi = ue_kpm[0]
            cell_id = ue_kpm[1]
            state = self.cells_states.get(cell_id, ())
            
            # Get latency for this UE from the latency map
            lat_data = latency_map.get(imsi, {'cell_avg': 0.0, 'ue_latency': 0.0})
            cell_avg_latency = lat_data['cell_avg']
            ue_latency = lat_data['ue_latency']
            
            # Build complete KPM tuple with latency
            new_ue_kpm = ue_kpm + (cell_avg_latency, ue_latency, state)
            ue_complete_kpms.append(new_ue_kpm)
        
        # Build DataFrame
        columns = (
            ["ueImsiComplete", "nrCellId"]
            + [
                "QosFlow.PdcpPduVolumeDL_Filter",
                "TB.TotNbrDl.1",
                "L3 serving SINR",
                "RRU.PrbUsedDl",
                "TB.TotNbrDlInitial.64Qam",
                "TB.TotNbrDlInitial.Qpsk",
                "TB.TotNbrDlInitial.16Qam",
            ]
            + [
                "DRB.PdcpSduDelayDl(cellAverageLatency)",
                "DRB.PdcpSduDelayDl.UEID (pdcpLatency)",
                "state"
            ]
        )
        
        df = pd.DataFrame(
            ue_complete_kpms,
            columns=columns[:len(ue_complete_kpms[0])] if ue_complete_kpms else columns
        )
        
        df["timestamp"] = self.last_timestamp
        df, _ = self.getRLFCounter(df, columns)
        df = self.ue_centric_tocell_centric(df)
        self.observations = self.offline_training_preprocessing(df)
        
        # Ensure all state columns exist
        for col in self.columns_state:
            if col not in self.observations.columns:
                self.observations[col] = 0.0
        
        # Log latency for debugging
        try:
            per_cell_lat = [
                float(self.observations.get(f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{c}", pd.Series([0.0])).iloc[0])
                for c in self.cellList
            ]
            sum_lat = float(self.observations.get("SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)", pd.Series([0.0])).iloc[0])
            self.log.info(
                "Latency per-cell=%s | SUM=%.3f ms | ts=%s",
                [round(x, 3) for x in per_cell_lat],
                sum_lat,
                int(self.last_timestamp),
            )
        except Exception as e:
            self.log.warning("Latency logging failed: %r", e)
        
        states = self.observations[self.columns_state]
        return np.asarray(states.iloc[0].values, dtype=np.float32)

    @override
    def _compute_reward(self):
        """
        Compute multi-objective reward with latency penalty.
        """
        cell_df = self.observations[self.columns_reward].copy()
        
        # Extract latency from aggregated column
        latency_ms = 0.0
        sum_lat = 0.0
        
        if 'SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)' in self.observations.columns:
            sum_lat = float(self.observations['SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)'].iloc[0])
            latency_ms = sum_lat
        
        # Normalize latency (0-1000ms range)
        latency_normalized = latency_ms / 1000.0
        
        # Multi-objective reward calculation
        throughput_term = 0.5 * (1 + cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter'].iloc[0])
        energy_term = -0.19 * (cell_df['SUM_TB.TotNbrDl.1'].iloc[0] * cell_df['ZERO_COUNT'].iloc[0])
        rlf_term = -0.2 * cell_df['SUM_RLF_VALUE'].iloc[0]
        oncost_term = -0.1 * cell_df['SUM_ES_ON_COST'].iloc[0]
        latency_term = -0.15 * latency_normalized  # Latency penalty
        
        reward = throughput_term + energy_term + rlf_term + oncost_term + latency_term
        
        # Log reward components
        self.log.info(
            "REWARD | thr=%.3f en=%.3f rlf=%.3f cost=%.3f lat=%.6f (%.3fms) | total=%.3f | step=%d",
            throughput_term, energy_term, rlf_term, oncost_term, latency_term, latency_ms,
            reward, self.num_steps
        )
        
        # Prepare Grafana database row
        db_row = {
            'timestamp': int(self.last_timestamp),
            'ueImsiComplete': None,
            'time_grafana': int(self.last_timestamp),
            'step': int(self.num_steps),
            'throughput': float(cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter'].iloc[0]) * 10 / 10**6,
            'en_cons': float(cell_df['SUM_TB.TotNbrDl.1'].iloc[0]),
            'rlf': float(cell_df['SUM_RLF_VALUE'].iloc[0]),
            'on_cost': float(cell_df['SUM_ES_ON_COST'].iloc[0]),
            'latency_cell_ms': float(latency_ms),
            'latency_ue_ms': float(latency_ms),
            'reward': float(reward)
        }
        
        # Insert into Grafana table
        self.datalake.insert_data("grafana", db_row)
        
        # Store per-cell latencies
        self._store_latency_in_sqlite(latency_ms, sum_lat)
        
        # Append to CSV
        self._append_qos_snapshot(latency_ms, latency_ms, reward)
        
        return reward

    def _store_latency_in_sqlite(self, avg_latency: float, sum_latency: float):
        """
        Store per-cell latency metrics in dedicated SQLite table.
        """
        try:
            latency_dict = {
                "timestamp": int(self.last_timestamp),
                "ueImsiComplete": None,
                "step": int(self.num_steps),
            }

            per_cell_latencies = []
            for cell in self.cellList:
                col_name = f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}"
                if col_name in self.observations.columns:
                    lat_val = float(self.observations[col_name].iloc[0])
                else:
                    lat_val = 0.0
                latency_dict[f"cell_{cell}_latency"] = lat_val
                per_cell_latencies.append(lat_val)

            latency_dict["avg_cell_latency"] = float(avg_latency)
            latency_dict["sum_latency"] = float(sum_latency)

            non_zero_latencies = [lat for lat in per_cell_latencies if lat > 0]
            latency_dict["max_latency"] = float(max(non_zero_latencies)) if non_zero_latencies else 0.0
            latency_dict["min_latency"] = float(min(non_zero_latencies)) if non_zero_latencies else 0.0

            self.datalake.insert_data("latency_tracking", latency_dict)

        except Exception as e:
            self.log.error("Failed to store latency: %r", e)

    @override
    def _compute_action(self, action):
        ts = int(self.last_timestamp % 2_000_000_000)
        cell_ids = self.cellList

        if self.heur:
            bits = np.asarray(action, dtype=int).flatten()
            if bits.size != len(cell_ids):
                raise ValueError(f"Invalid action size: {bits.size} != {len(cell_ids)}")
            bits = (bits > 0).astype(int)
        else:
            dec_action = self.action_list[int(action)]
            bits = np.array([(dec_action >> i) & 1 for i in range(len(cell_ids))], dtype=int)

        bits = np.array([1 if b == 0 else 0 if b == 1 else b for b in bits], dtype=int)

        actions = []
        for i, cell in enumerate(cell_ids):
            ho_allowed = int(bits[i])
            actions.append((ts, int(cell), ho_allowed))
            self.datalake.insert_data(
                "bsState",
                {
                    "timestamp": ts,
                    "ueImsiComplete": None,
                    "cellId": int(cell),
                    "state": ho_allowed,
                },
            )

        self.previous_inverted_action = "".join("1" if b == 0 else "0" for b in bits.tolist())

        return actions

    @override
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
            "reward": "REAL"
        }
        
        self.datalake._create_table("bsState", self.gnb_state_keys)
        self.datalake._create_table("grafana", grafana_keys)
        self.datalake._create_table("latency_tracking", self.latency_tracking_keys)
        
        return super()._init_datalake_usecase()

    @override
    def _fill_datalake_usecase(self):
        for file_path in glob.glob(os.path.join(self.sim_path, "bsState.txt")):
            with open(file_path, "r") as csvfile:
                for row in csv.DictReader(csvfile, delimiter=" "):
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

    def ue_centric_tocell_centric(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ueImsiComplete" in df.columns:
            df.drop(columns=["ueImsiComplete"], inplace=True)

        if "L3 serving SINR" in df.columns:
            df["L3 serving SINR"] = df["L3 serving SINR"].replace(-np.inf, 0)
            df.drop(columns=["L3 serving SINR"], inplace=True)

        df = df.drop_duplicates()
        df.reset_index(drop=True, inplace=True)
        return df

    def rename_columns(self, columns, cell_no):
        return [f"{c}_{cell_no}" for c in columns]

    def offline_training_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_eekpi_qpsk_16_64qam_sum_and_ratio(df)
        df.sort_values(by=["timestamp"], ascending=True, inplace=True)

        if "state" in df.columns:
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

            # Handle latency column aliases
            alias_cols = [
                f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}",
                f"PdcpSduDelayDl.UEID (pdcpLatency)_{cell}",
                f"DRB.PdcpSduDelayDl(UEID)_{cell}",
                f"PdcpSduDelayDl(UEID)_{cell}",
                f"DRB.PdcpSduDelayDl(cellAverageLatency)_{cell}",
                f"PdcpSduDelayDl(cellAverageLatency)_{cell}",
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

        # Aggregate metrics including latency
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

        cell_df["ACTION_BINARY"] = self.previous_inverted_action
        cell_df["ACTION_BINARY"] = cell_df["ACTION_BINARY"].astype(str)
        cell_df["ZERO_COUNT"] = cell_df["ACTION_BINARY"].apply(lambda x: x.count("0"))

        return cell_df

    def add_eekpi_qpsk_16_64qam_sum_and_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        if {
            "TB.TotNbrDlInitial.Qpsk",
            "TB.TotNbrDlInitial.16Qam",
            "TB.TotNbrDlInitial.64Qam",
        }.issubset(df.columns):
            df["TB.TOTNBRDLINITIAL.SUM"] = (
                df["TB.TotNbrDlInitial.Qpsk"]
                + df["TB.TotNbrDlInitial.16Qam"]
                + df["TB.TotNbrDlInitial.64Qam"]
            )

            df["TB_TOTNBRDLINITIAL_64QAM_RATIO"] = (
                df["TB.TotNbrDlInitial.64Qam"] / df["TB.TOTNBRDLINITIAL.SUM"]
            ).replace([np.inf, -np.inf], 0).fillna(0.00001)

        if "RRU.PrbUsedDl" in df.columns:
            df["RRU.PrbUsedDl"] = df["RRU.PrbUsedDl"].replace(0, 0.00001)

        if "TB.TotNbrDl.1" in df.columns:
            df["TB.TotNbrDl.1"] = df["TB.TotNbrDl.1"].replace(0, 0.00001)

        return df

    def getRLFCounter(self, df: pd.DataFrame, columns):
        df["timestamp"] = df["timestamp"].astype(int)
        df["RLF_Counter"] = 0.0
        df["RLF_VALUE"] = 0

        if "L3 serving SINR" in df.columns:
            df["L3 serving SINR"] = df["L3 serving SINR"].replace(-np.inf, 0)

        grouped = df.groupby(["timestamp", "nrCellId"])

        for (timestamp, cell), group in grouped:
            total_count = group.shape[0]
            if total_count > 0 and "L3 serving SINR" in group.columns:
                num_values = group[group["L3 serving SINR"] < -5].shape[0]
                rlf_value = (num_values / total_count) * 100
                mask = (df["timestamp"] == timestamp) & (df["nrCellId"] == cell)
                df.loc[mask, "RLF_Counter"] = rlf_value
                df.loc[mask, "RLF_VALUE"] = num_values

        return df, columns

    def es_on_cost_calculation(self, cell_df: pd.DataFrame) -> pd.DataFrame:
        for cell in self.cellList:
            current_timestamp = self.last_timestamp
            time_diff_obs = []

            current_state = self.cells_states.get(cell, ())
            current_state = 1 if current_state == 0 else 0

            if current_state == 1:
                if self.cell_timestamp_state_dict[cell] == float("inf"):
                    time_diff_obs.append(100)
                    self.cell_timestamp_state_dict[cell] = current_timestamp
                else:
                    time_diff_obs.append(current_timestamp - self.cell_timestamp_state_dict[cell] + 100)
            else:
                time_diff_obs.append(float("inf"))
                self.cell_timestamp_state_dict[cell] = float("inf")

            time_diff_obs_col = f"TIME_DIFF_OBS_{cell}"
            cell_df[time_diff_obs_col] = time_diff_obs

            es_on_cost_col = f"ES_ON_COST_{cell}"
            cell_df[es_on_cost_col] = cell_df[time_diff_obs_col].apply(
                lambda diff: self.Cf * ((1 - self.lambdaf) ** (diff * self.time_factor))
                if diff != float("inf")
                else 0
            )

        return cell_df

    def _update_cell_states(self):
        cell_states_table = self.datalake.read_table("bsState")
        states_of_interest = []

        for cell_state in cell_states_table:
            if cell_state[0] == (self.last_timestamp - 100):
                states_of_interest.append(cell_state)

        if len(states_of_interest) != len(self.cellList):
            for cellId in self.cellList:
                self.cells_states[cellId] = 1
        else:
            for state in states_of_interest:
                cellId = state[2]
                self.cells_states[cellId] = state[3]

    def bs_states_list(self):
        cell_states_table = self.datalake.read_table("bsState")
        states_of_interest = []

        for cell_state in cell_states_table:
            if cell_state[0] == self.last_timestamp:
                states_of_interest.append(cell_state)

        current_kpms = [state[3] for state in states_of_interest]
        inverted_action_ar = [1 if element == 0 else 0 for element in current_kpms]

        return inverted_action_ar

    def get_latency_history(self):
        """
        Query all latency tracking data from SQLite database.
        """
        try:
            return self.datalake.read_table("latency_tracking")
        except Exception as e:
            self.log.error("Failed to read latency history: %r", e)
            return []
