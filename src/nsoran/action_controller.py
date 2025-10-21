from __future__ import annotations

import os
from os import path
from typing import Iterable, Tuple, Any


class ActionController:
    """
    The ActionController class is responsible for delivering the action to ns-O-RAN.
    In the stand-alone mode, the action is delivered by writing on the appropriate file.
    """

    directory: str
    log_filename: str
    control_filename: str

    def __init__(self, sim_path: str, log_filename: str, control_filename: str, header: Iterable[str]):
        """Initialize Controller and its files
        Args:
            sim_path (str): the simulation path
            log_filename (str): the name of the file where the; This file is purely for logging purposed and it is not read by ns-3
            control_filename (str): the name of the control file that delivers the action to ns-3; This file is read by ns-3
            header (dict|iterable): fields of the action the agent is going to write
        """
        self.directory = sim_path
        self.log_filename = log_filename
        self.control_filename = control_filename

        # Garante que o diretório exista
        if not path.isdir(self.directory):
            os.makedirs(self.directory, exist_ok=True)

        # Cria/zera o arquivo de log com cabeçalho
        log_path = path.join(self.directory, self.log_filename)
        with open(log_path, "w") as file:
            file.write(f"{','.join(header)}\n")
            file.flush()
            os.fsync(file.fileno())

        # Toca/cria o arquivo de controle (o ns-3 vai ler este arquivo)
        ctl_path = path.join(self.directory, self.control_filename)
        open(ctl_path, "a").close()

    def create_control_action(self, timestamp: int, actions: Iterable[Tuple[Any, ...]]):
        """Applies the control action by writing it in the appropriate file
            timestamp (int) : action's timestamp
            actions [(tuple)]: list of tuples representing the actions to be sent
        """
        log_path = path.join(self.directory, self.log_filename)
        ctl_path = path.join(self.directory, self.control_filename)

        # Abre ambos e grava linhas espelhadas
        with open(log_path, "a") as logFile, open(ctl_path, "a") as file:
            for action in actions:
                # Linha CSV: timestamp,actionType,id,value
                control_action = f"{timestamp},{','.join(map(str, action))}\n"
                file.write(control_action)
                logFile.write(control_action)

            # Flush dos buffers Python
            file.flush()
            logFile.flush()

            # MUITO IMPORTANTE: fsync garante persistência antes do ns-3 ler
            os.fsync(file.fileno())
            os.fsync(logFile.fileno())
