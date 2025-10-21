from collections import defaultdict
import os
import sqlite3
import re

class SQLiteDatabaseAPI:
    lte_cu_cp_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "numActiveUes": "INTEGER",
        "cellId": "INTEGER",
        "DRB.EstabSucc.5QI.UEID (numDrb)": "INTEGER",
        "sameCellSinr": "REAL",
        "sameCellSinr 3gpp encoded": "REAL"
    }
    gnb_cu_cp_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "cellId": "INTEGER",
        "numActiveUes": "INTEGER",
        "DRB.EstabSucc.5QI.UEID (numDrb)": "INTEGER",
        "L3 serving Id(m_cellId)": "INTEGER",
        "UE (imsi)": "INTEGER",
        "L3 serving SINR": "REAL",
        "L3 serving SINR 3gpp": "REAL",
        "L3 neigh Id 1 (cellId)": "INTEGER",
        "L3 neigh SINR 1": "REAL",
        "L3 neigh SINR 3gpp 1 (convertedSinr)": "REAL",
        "L3 neigh Id 2 (cellId)": "INTEGER",
        "L3 neigh SINR 2": "REAL",
        "L3 neigh SINR 3gpp 2 (convertedSinr)": "REAL",
        "L3 neigh Id 3 (cellId)": "INTEGER",
        "L3 neigh SINR 3": "REAL",
        "L3 neigh SINR 3gpp 3 (convertedSinr)": "REAL",
        "L3 neigh Id 4 (cellId)": "INTEGER",
        "L3 neigh SINR 4": "REAL",
        "L3 neigh SINR 3gpp 4 (convertedSinr)": "REAL",
        "L3 neigh Id 5 (cellId)": "INTEGER",
        "L3 neigh SINR 5": "REAL",
        "L3 neigh SINR 3gpp 5 (convertedSinr)": "REAL",
        "L3 neigh Id 6 (cellId)": "INTEGER",
        "L3 neigh SINR 6": "REAL",
        "L3 neigh SINR 3gpp 6 (convertedSinr)": "REAL"
    }
    lte_cu_up_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "cellId": "INTEGER",
        "DRB.PdcpSduDelayDl(cellAverageLatency)": "REAL",
        "m_pDCPBytesDL(cellDlTxVolume)": "REAL",
        "DRB.PdcpSduVolumeDl_Filter.UEID (txBytes)": "REAL",
        "Tot.PdcpSduNbrDl.UEID (txDlPackets)": "REAL",
        "DRB.PdcpSduBitRateDl.UEID (pdcpThroughput)": "REAL",
        "DRB.PdcpSduDelayDl.UEID (pdcpLatency)": "REAL",
    }
    gnb_cu_up_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "cellId": "INTEGER",
        "QosFlow.PdcpPduVolumeDL_Filter.UEID(txPdcpPduBytesNrRlc)": "REAL",
        "DRB.PdcpPduNbrDl.Qos.UEID (txPdcpPduNrRlc)": "REAL"
    }
    du_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "nrCellId": "INTEGER",
        "dlAvailablePrbs": "REAL",
        "ulAvailablePrbs": "REAL",
        "qci": "INTEGER",
        "dlPrbUsage": "REAL",
        "ulPrbUsage": "REAL",
        "TB.TotNbrDl.1": "REAL",
        "TB.TotNbrDlInitial": "REAL",
        "TB.TotNbrDlInitial.Qpsk": "REAL",
        "TB.TotNbrDlInitial.16Qam": "REAL",
        "TB.TotNbrDlInitial.64Qam": "REAL",
        "RRU.PrbUsedDl": "REAL",
        "TB.ErrTotalNbrDl.1": "REAL",
        "QosFlow.PdcpPduVolumeDL_Filter": "REAL",
        "CARR.PDSCHMCSDist.Bin1": "REAL",
        "CARR.PDSCHMCSDist.Bin2": "REAL",
        "CARR.PDSCHMCSDist.Bin3": "REAL",
        "CARR.PDSCHMCSDist.Bin4": "REAL",
        "CARR.PDSCHMCSDist.Bin5": "REAL",
        "CARR.PDSCHMCSDist.Bin6": "REAL",
        "L1M.RS-SINR.Bin34": "REAL",
        "L1M.RS-SINR.Bin46": "REAL",
        "L1M.RS-SINR.Bin58": "REAL",
        "L1M.RS-SINR.Bin70": "REAL",
        "L1M.RS-SINR.Bin82": "REAL",
        "L1M.RS-SINR.Bin94": "REAL",
        "L1M.RS-SINR.Bin127": "REAL",
        "DRB.BufferSize.Qos": "REAL",
        "DRB.MeanActiveUeDl": "REAL",
        "TB.TotNbrDl.1.UEID": "REAL",
        "TB.TotNbrDlInitial.UEID": "REAL",
        "TB.TotNbrDlInitial.Qpsk.UEID": "REAL",
        "TB.TotNbrDlInitial.16Qam.UEID": "REAL",
        "TB.TotNbrDlInitial.64Qam.UEID": "REAL",
        "TB.ErrTotalNbrDl.1.UEID": "REAL",
        "QosFlow.PdcpPduVolumeDL_Filter.UEID": "REAL",
        "RRU.PrbUsedDl.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin1.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin2.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin3.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin4.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin5.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin6.UEID": "REAL",
        "L1M.RS-SINR.Bin34.UEID": "REAL",
        "L1M.RS-SINR.Bin46.UEID": "REAL",
        "L1M.RS-SINR.Bin58.UEID": "REAL",
        "L1M.RS-SINR.Bin70.UEID": "REAL",
        "L1M.RS-SINR.Bin82.UEID": "REAL",
        "L1M.RS-SINR.Bin94.UEID": "REAL",
        "L1M.RS-SINR.Bin127.UEID": "REAL",
        "DRB.BufferSize.Qos.UEID": "REAL",
        "DRB.UEThpDl.UEID": "REAL",
        "DRB.UEThpDlPdcpBased.UEID": "REAL"
    }

    debug: bool = False

    def __init__(self, simulation_dir, num_ues_gnb, debug=False):
        self.simulation_dir = simulation_dir
        self.num_ues = num_ues_gnb * 7  # number of gNBs in the scenario
        self.database_path = os.path.join(simulation_dir, 'database.db')
        self.tables = {}  # table_name -> {kpm: type}
        self.debug = debug

        self.acquire_connection()
        self._create_table("lte_cu_cp", self.lte_cu_cp_keys)
        self._create_table("gnb_cu_cp", self.gnb_cu_cp_keys)
        self._create_table("lte_cu_up", self.lte_cu_up_keys)
        self._create_table("gnb_cu_up", self.gnb_cu_up_keys)
        self._create_table("du", self.du_keys)
        self.release_connection()

    @staticmethod
    def sanitize_column_name(column_name):
        column_name = str(column_name).lower()
        column_name = column_name.replace(' ', '_')
        column_name = re.sub(r'[^\w\s]', '', column_name)
        return column_name

    def acquire_connection(self):
        self.connection = sqlite3.connect(self.database_path)
        if self.debug:
            self.connection.set_trace_callback(print)
        self.cursor = self.connection.cursor()
        return True

    def release_connection(self):
        if getattr(self, "connection", None) is None:
            return True
        self.connection.commit()
        self.connection.close()
        self.connection = None
        return True

    def lock_connection(func):
        def wrapper(self, *args, **kwargs):
            need_connection = getattr(self, "connection", None) is None
            if need_connection:
                self.acquire_connection()
            if self.debug:
                self.connection.set_trace_callback(print)
            result = func(self, *args, **kwargs)
            if need_connection:
                self.release_connection()
            return result
        return wrapper

    @lock_connection
    def _create_table(self, table_name: str, columns: dict[str, str]):
        if getattr(self, "connection", None) is None:
            print(f"Error in creating table {table_name}: Not connected to the database.")
            return

        column_definitions = ', '.join(
            [f"{SQLiteDatabaseAPI.sanitize_column_name(name)} {typ}" for name, typ in columns.items()]
        )
        column_definitions += f", UNIQUE (timestamp, {SQLiteDatabaseAPI.sanitize_column_name('ueImsiComplete')})"
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})"
        self.cursor.execute(query)
        self.tables[table_name] = columns
        if self.debug:
            print(f"Table '{table_name}' created or exists.")

    @lock_connection
    def entry_exists(self, table_name, timestamp, ue_imsi_complete) -> bool:
        query = (
            f"SELECT COUNT(*) FROM {table_name} "
            f"WHERE {SQLiteDatabaseAPI.sanitize_column_name('timestamp')} = ? "
            f"AND {SQLiteDatabaseAPI.sanitize_column_name('ueImsiComplete')} = ?"
        )
        values = (timestamp, ue_imsi_complete)
        result = self.cursor.execute(query, values).fetchone()
        return result[0] > 0

    def insert_lte_cu_cp(self, data: dict): self.insert_data('lte_cu_cp', data)
    def insert_gnb_cu_cp(self, data: dict): self.insert_data('gnb_cu_cp', data)
    def insert_lte_cu_up(self, data: dict): self.insert_data('lte_cu_up', data)
    def insert_gnb_cu_up(self, data: dict): self.insert_data('gnb_cu_up', data)
    def insert_du(self, data: dict): self.insert_data('du', data)

    @lock_connection
    def insert_data(self, table_name, data: dict):
        if table_name not in self.tables:
            raise ValueError(f'Input table name not found: {table_name} not in {self.tables.keys()}')

        admitted_keys = self.tables[table_name]
        filtered_kpms = {key: value for key, value in data.items() if key in admitted_keys}
        if not filtered_kpms:
            raise ValueError("No acceptable columns found in the input dictionary.")

        if 'timestamp' not in filtered_kpms or 'ueImsiComplete' not in filtered_kpms:
            raise ValueError("Missing required keys: 'timestamp' and 'ueImsiComplete'.")

        if self.entry_exists(table_name, filtered_kpms['timestamp'], filtered_kpms['ueImsiComplete']):
            return

        placeholders = ', '.join(['?' for _ in filtered_kpms.values()])
        columns = ', '.join([SQLiteDatabaseAPI.sanitize_column_name(col_name) for col_name in filtered_kpms.keys()])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        values = tuple(filtered_kpms.values())
        self.cursor.execute(query, values)
        if self.debug:
            print(f"Data inserted into '{table_name}'.")

    @lock_connection
    def read_table(self, table_name):
        query = f"SELECT * FROM {table_name}"
        result = self.cursor.execute(query)
        return result.fetchall()

    @lock_connection
    def read_kpms(self, timestamp: int, required_kpms: list) -> list[tuple]:
        """
        Query the datalake to retrieve the observation vector.

        Retorna SEMPRE list[tuple]. A primeira coluna do resultado é o ueImsiComplete.
        """
        # 0) evita duplicar IMSI caso venha no required_kpms
        if required_kpms is None:
            required_kpms = []
        required_kpms = [k for k in required_kpms if str(k).lower() != 'ueimsicomplete' and k != 'ueImsiComplete']

        # 1) mapeia onde cada KPM existe (usando nomes originais)
        tables_involved: dict[list] = {}
        found = [False] * len(required_kpms)

        for table_name, keys in self.tables.items():
            for i, kpm in enumerate(required_kpms):
                if kpm in keys:
                    tables_involved.setdefault(table_name, []).append(kpm)
                    found[i] = True

        not_found = [k for ok, k in zip(found, required_kpms) if not ok]
        if not_found:
            raise ValueError(f"Column(s) {not_found} not found in any table.")

        if not tables_involved:
            return []

        # 2) SELECT/JOINS preservando a ordem de required_kpms
        from_clause = next(iter(tables_involved))
        ts_col = self.sanitize_column_name("timestamp")
        ue_col = self.sanitize_column_name("ueImsiComplete")

        select_clause = [f"{from_clause}.{ue_col} AS ueImsiComplete"]  # IMSI sempre primeiro
        join_clause = []

        kpm_to_tables = defaultdict(list)
        for t, kpms in tables_involved.items():
            for k in kpms:
                kpm_to_tables[k].append(t)

        for kpm in required_kpms:
            kpm_san = self.sanitize_column_name(kpm)
            tables_for = kpm_to_tables[kpm]
            if len(tables_for) > 1:
                for t in tables_for:
                    select_clause.append(f"{t}.{kpm_san} AS {kpm_san}_{t}")
            else:
                t = tables_for[0]
                select_clause.append(f"{t}.{kpm_san}")

        # JOIN por timestamp e IMSI
        tables = list(tables_involved.keys())
        base = tables[0]
        for t in tables[1:]:
            join_clause.append(
                f"INNER JOIN {t} ON {base}.{ts_col} = {t}.{ts_col} AND {base}.{ue_col} = {t}.{ue_col}"
            )

        query = f"SELECT {', '.join(select_clause)} FROM {from_clause}"
        if join_clause:
            query += " " + " ".join(join_clause)
        query += f" WHERE {from_clause}.{ts_col} = ?"

        # 3) Executa e normaliza (sempre list[tuple])
        try:
            rows = self.cursor.execute(query, (timestamp,)).fetchall() or []
        except sqlite3.Error as e:
            raise RuntimeError(f"SQLite error on read_kpms: {e}\nQuery: {query}")

        return [r if isinstance(r, tuple) else (r,) for r in rows]

    @staticmethod
    def extract_cellId(filepath) -> int:
        pattern = r'(\d+)\.txt$'
        match = re.search(pattern, filepath)
        if match:
            return int(match.group(1))
        raise ValueError("Unable to extract cellId")

    def __del__(self):
        if getattr(self, "connection", None) is not None:
            self.release_connection()
            if self.debug:
                print("Connection to the database closed.")
