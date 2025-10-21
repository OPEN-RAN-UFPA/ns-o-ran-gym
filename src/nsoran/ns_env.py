from abc import abstractmethod
import fcntl
import selectors
import time
from typing import Any, SupportsFloat, Tuple, Dict
import uuid
import gymnasium as gym
import os
import glob
import csv
from posix_ipc import Semaphore, O_CREAT, BusyError
from .action_controller import ActionController
from .datalake import SQLiteDatabaseAPI
from importlib.machinery import SourceFileLoader
import types
import subprocess
import sem

class NsOranEnv(gym.Env):
    """Base class for an ns-O-RAN environment compliant with Gymnasium."""
    metadata = {'render_modes': ['ansi']}

    ns3_path: str
    scenario: str
    scenario_configuration: dict
    output_folder: str
    optimized: bool
    skip_configuration: bool
    sim_path: str
    sim_process: subprocess.Popen
    metricsReadySemaphore: Semaphore
    controlSemaphore: Semaphore
    control_header: list
    log_file: str
    control_file: str
    is_open: bool
    action_controller: ActionController
    datalake: SQLiteDatabaseAPI

    def __init__(
        self,
        render_mode: str = None,
        ns3_path: str = None,
        scenario: str = None,
        scenario_configuration: dict = None,
        output_folder: str = None,
        optimized: bool = True,
        skip_configuration: bool = False,
        control_header: list = [],
        log_file: str = '',
        control_file: str = ''
    ):
        if render_mode and render_mode not in self.metadata['render_modes']:
            raise ValueError(f'{render_mode} is not a valid render mode. Values accepted are: {self.metadata["render_modes"]}')
        self.render_mode = render_mode

        self.ns3_path = ns3_path
        self.scenario = scenario
        # flatten config dict (expects values as lists)
        self.scenario_configuration = {k: v[0] for k, v in scenario_configuration.items() if v}
        self.output_folder = output_folder
        self.optimized = optimized
        self.skip_configuration = skip_configuration
        self.control_header = control_header
        self.log_file = log_file
        self.control_file = control_file

        self.is_open = False
        self.return_info = False
        self.last_timestamp = 0
        self.terminated = False
        self.truncated = False

        self.setup_sim()

    # --------------------------
    # ns-3 setup / build
    # --------------------------
    def setup_sim(self):
        """Configure paths and find scenario executable."""
        if self.optimized:
            library_path = "%s:%s" % (
                os.path.join(self.ns3_path, 'build/optimized'),
                os.path.join(self.ns3_path, 'build/optimized/lib'))
        else:
            library_path = "%s:%s" % (
                os.path.join(self.ns3_path, 'build/'),
                os.path.join(self.ns3_path, 'build/lib'))

        self.environment = {'LD_LIBRARY_PATH': library_path, 'DYLD_LIBRARY_PATH': library_path}

        # Configure and build ns-3
        self.configure_and_build_ns3()

        # Determine build status file / module to read runnable programs
        if os.path.exists(os.path.join(self.ns3_path, "ns3")):
            build_status_fname = ".lock-ns3_%s_build" % os.sys.platform
            build_status_path = os.path.join(self.ns3_path, build_status_fname)
        else:
            build_status_fname = "build.py"
            if self.optimized:
                build_status_path = os.path.join(self.ns3_path, 'build/optimized/build-status.py')
            else:
                build_status_path = os.path.join(self.ns3_path, 'build/build-status.py')

        loader = SourceFileLoader(build_status_fname, build_status_path)
        mod = types.ModuleType(loader.name)
        loader.exec_module(mod)

        matches = [{'name': program, 'path': os.path.abspath(os.path.join(self.ns3_path, program))}
                   for program in mod.ns3_runnable_programs if self.scenario in program]
        if not matches:
            raise ValueError(f"Cannot find {self.scenario} script")

        match_percentages = map(lambda x: {'name': x['name'],
                                           'path': x['path'],
                                           'percentage': len(self.scenario) / len(x['name'])},
                                matches)
        self.script_executable = max(match_percentages, key=lambda x: x['percentage'])['path']

        # For non-CMake builds with scratch path
        if "scratch" in self.script_executable and not os.path.exists(os.path.join(self.ns3_path, "ns3")):
            path_with_subdir = self.script_executable.split("/scratch/")[-1]
            if "/" in path_with_subdir:
                executable_subpath = "%s/%s" % (self.scenario, self.scenario)
            else:
                executable_subpath = self.scenario
            if self.optimized:
                self.script_executable = os.path.abspath(
                    os.path.join(self.ns3_path, "build/optimized/scratch", executable_subpath))
            else:
                self.script_executable = os.path.abspath(
                    os.path.join(self.ns3_path, "build/scratch", executable_subpath))

    def configure_and_build_ns3(self):
        """Configure and build the ns-3 code."""
        build_program = "./ns3" if os.path.exists(os.path.join(self.ns3_path, "ns3")) else "./waf"

        if not self.skip_configuration:
            configuration_command = ['python3', build_program, 'configure', '--enable-examples', '--disable-gtk', '--disable-werror']
            if self.optimized:
                configuration_command += ['--build-profile=optimized', '--out=build/optimized']
            subprocess.run(configuration_command, cwd=self.ns3_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        j_argument = ['-j', str(os.cpu_count())]
        subprocess.run(['python3', build_program] + j_argument + ['build'], cwd=self.ns3_path,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # --------------------------
    # Simulation lifecycle
    # --------------------------
    def start_sim(self):
        """Start simulation + controllers + semaphores."""
        if self.is_open:
            raise ValueError('The environment is open and a new start_sim has been called.')

        self.is_open = True
        parameters = self.scenario_configuration

        self.sim_result = {'params': {}, 'meta': {}}
        self.sim_result['params'].update(parameters)

        command = [self.script_executable] + [f'--{param}={value}' for param, value in parameters.items()]

        sim_uuid = str(uuid.uuid4())
        self.sim_result['meta']['id'] = sim_uuid
        self.sim_path = os.path.join(self.output_folder, self.sim_result['meta']['id'])
        os.makedirs(self.sim_path, exist_ok=True)

        # Prepare stdout/stderr files (create empty if missing)
        open(os.path.join(self.sim_path, 'stdout'), 'a').close()
        open(os.path.join(self.sim_path, 'stderr'), 'a').close()

        # Create Datalake and Action Controller
        if not self.log_file:
            raise ValueError('Missing the log file path.')
        if not self.control_file:
            raise ValueError('Missing the control file path.')
        if not self.control_header:
            raise ValueError('Missing the list of values to perform control.')

        self.action_controller = ActionController(self.sim_path, self.log_file, self.control_file, self.control_header)
        self.datalake = SQLiteDatabaseAPI(self.sim_path, num_ues_gnb=self.sim_result['params']['ues'], debug=False)
        self._init_datalake_usecase()

        # Semaphores
        nameMetricsReadySemaphore = "/sem_metrics_" + self.sim_path.split('/')[-1]
        nameControlSemaphore = "/sem_control_" + self.sim_path.split('/')[-1]
        self.metricsReadySemaphore = Semaphore(nameMetricsReadySemaphore, O_CREAT, mode=0o660, initial_value=0)
        self.controlSemaphore = Semaphore(nameControlSemaphore, O_CREAT, mode=0o660, initial_value=0)
        self.last_timestamp = 0

        # Launch simulation
        self.sim_result['meta']['start_time'] = time.time()
        self.sim_process = subprocess.Popen(
            command, cwd=self.sim_path, env=self.environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Non-blocking stdout/stderr
        self.selector = selectors.DefaultSelector()
        self._set_nonblocking(self.sim_process.stdout)
        self._set_nonblocking(self.sim_process.stderr)
        self.selector.register(self.sim_process.stdout, selectors.EVENT_READ)
        self.selector.register(self.sim_process.stderr, selectors.EVENT_READ)

    @staticmethod
    def _set_nonblocking(fileobj):
        """Set non-blocking mode on a file object."""
        fd = fileobj.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def read_streams(self):
        """Drain stdout/stderr without blocking. Append to files (no truncation)."""
        events = self.selector.select(timeout=0)
        stdout_file_path = os.path.join(self.sim_path, 'stdout')
        stderr_file_path = os.path.join(self.sim_path, 'stderr')

        if not events:
            return

        with open(stdout_file_path, 'a') as stdout_file, open(stderr_file_path, 'a') as stderr_file:
            for key, _ in events:
                try:
                    data = key.fileobj.read()
                except Exception:
                    data = ''
                if not data:
                    continue
                if key.fileobj is self.sim_process.stdout:
                    stdout_file.write(data)
                elif key.fileobj is self.sim_process.stderr:
                    stderr_file.write(data)

    def is_simulation_over(self) -> bool:
        """Check whether the simulation is over; set terminated/truncated flags accordingly."""
        self.read_streams()
        if self.sim_process.poll() is None:
            return False

        # Ensure any remaining output is processed
        self.read_streams()

        end = time.time()
        return_code = self.sim_process.returncode

        if return_code != 0:
            # The environment should return truncated.
            self.terminated = False
            self.truncated = True

            stdout_file_path = os.path.join(self.sim_path, 'stdout')
            stderr_file_path = os.path.join(self.sim_path, 'stderr')
            with open(stdout_file_path, 'r') as stdout_file, open(stderr_file_path, 'r') as stderr_file:
                complete_command = sem.utils.get_command_from_result(self.scenario, self.sim_result)
                complete_command_debug = sem.utils.get_command_from_result(self.scenario, self.sim_result, debug=True)
                error_message = (
                    f'\nSimulation exited with an error.\nStderr: {stderr_file.read()}\n'
                    f'Stdout: {stdout_file.read()}\nUse this command to reproduce:\n{complete_command}\n'
                    f'Debug with gdb:\n{complete_command_debug}'
                )
                print(error_message)
        else:
            # time limit / natural end
            self.terminated = True
            self.truncated = True

        self.sim_result['meta']['elapsed_time'] = end - self.sim_result['meta']['start_time']
        self.sim_result['meta']['exitcode'] = return_code
        return True

    # --------------------------
    # Gymnasium API
    # --------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[object, Dict[str, Any]]:
        super().reset(seed=seed)
        self.close()
        self.start_sim()

        if options and 'return_info' in options:
            self.return_info = bool(options['return_info'])
        else:
            self.return_info = False

        # wait first metrics, load into datalake
        self._wait_data_availability(timeout=5)
        self._fill_datalake()

        self.terminated = False
        self.truncated = False

        obs = self._get_obs()
        info = self.render() if self.return_info else {}
        return obs, info

    def _wait_data_availability(self, timeout: int = 10):
        """Wait for metrics semaphore or end of simulation."""
        is_still_active = True
        while is_still_active:
            try:
                self.metricsReadySemaphore.acquire(timeout=timeout)
                break
            except BusyError:
                is_still_active = not self.is_simulation_over()

    def step(self, action: object) -> Tuple[object, SupportsFloat, bool, bool, Dict[str, Any]]:
        """One environment step: write action, signal control, wait metrics, load datalake, return (obs, reward,...)."""
        # If simulation still running, send action and wait for next metrics
        if not self.is_simulation_over():
            actions = self._compute_action(action)
            self.action_controller.create_control_action(self.last_timestamp, actions)
            self.controlSemaphore.release()
            self._wait_data_availability()
            self._fill_datalake()

        # Compose observation and reward once
        obs = self._get_obs()
        reward = self._compute_reward()

        # increment step counter if child uses it
        if hasattr(self, "num_steps"):
            try:
                self.num_steps += 1  # type: ignore[attr-defined]
            except Exception:
                pass

        info = self.render() if self.return_info else {'last_timestamp': self.last_timestamp}
        return obs, reward, self.terminated, self.truncated, info

    # --------------------------
    # Datalake ingestion
    # --------------------------
    def _fill_datalake(self):
        """Collect latest KPMs from CSV files and upload them to the Datalake."""
        self.datalake.acquire_connection()

        for file_path in glob.glob(os.path.join(self.sim_path, 'cu-up-cell-*.txt')):
            with open(file_path, 'r') as csvfile:
                for row in csv.DictReader(csvfile):
                    timestamp = int(row['timestamp'])
                    if timestamp >= self.last_timestamp:
                        cellId = self.datalake.extract_cellId(file_path)
                        row['cellId'] = cellId
                        if cellId == 1:
                            self.datalake.insert_lte_cu_up(row)
                        else:
                            self.datalake.insert_gnb_cu_up(row)
                        self.last_timestamp = timestamp

        for file_path in glob.glob(os.path.join(self.sim_path, 'cu-cp-cell-*.txt')):
            with open(file_path, 'r') as csvfile:
                for row in csv.DictReader(csvfile):
                    timestamp = int(row['timestamp'])
                    if timestamp >= self.last_timestamp:
                        cellId = self.datalake.extract_cellId(file_path)
                        row['cellId'] = cellId
                        if cellId == 1:
                            self.datalake.insert_lte_cu_cp(row)
                        else:
                            self.datalake.insert_gnb_cu_cp(row)
                        self.last_timestamp = timestamp

        for file_path in glob.glob(os.path.join(self.sim_path, 'du-cell-*.txt')):
            with open(file_path, 'r') as csvfile:
                for row in csv.DictReader(csvfile):
                    timestamp = int(row['timestamp'])
                    if timestamp >= self.last_timestamp:
                        self.datalake.insert_du(row)
                        self.last_timestamp = timestamp

        self._fill_datalake_usecase()
        self.datalake.release_connection()

    # --------------------------
    # Abstracts for use-cases
    # --------------------------
    @abstractmethod
    def _compute_action(self, action) -> list[tuple]:
        """Convert agent action into (timestamp, cellId, ...) tuples for ns-3 control."""
        raise NotImplementedError('_compute_action() must be implemented for the specific use case')

    @abstractmethod
    def _get_obs(self) -> object:
        """Return current observation for the specific use case."""
        raise NotImplementedError('_get_obs() must be implemented for the specific use case')

    @abstractmethod
    def _compute_reward(self):
        """Return current reward for the specific use case."""
        raise NotImplementedError('_compute_reward() must be implemented for the specific use case')

    @abstractmethod
    def _init_datalake_usecase(self):
        """Optional: initialize extra datalake resources for the use case."""
        pass

    @abstractmethod
    def _fill_datalake_usecase(self):
        """Optional: ingest extra data for the use case."""
        pass

    # --------------------------
    # Misc
    # --------------------------
    def _get_info(self):
        # could include more telemetry later
        return {'isopen': self.is_open, 'results': self.sim_result}

    def render(self):
        if self.render_mode == "ansi":
            infos = self._get_info()
            print(infos)
            return infos
        return {}

    def close(self):
        super().close()
        if not self.is_open:
            return
        # signal and terminate process safely
        try:
            # wake any waiter to allow exit
            try:
                self.metricsReadySemaphore.release()
            except Exception:
                pass
            try:
                self.controlSemaphore.release()
            except Exception:
                pass

            if self.sim_process and (self.sim_process.poll() is None):
                # give it a moment to exit cleanly
                try:
                    self.sim_process.terminate()
                    try:
                        self.sim_process.wait(timeout=3)
                    except Exception:
                        self.sim_process.kill()
                except Exception:
                    pass

            # unlink semaphores
            try:
                self.controlSemaphore.unlink()
            except Exception:
                pass
            try:
                self.metricsReadySemaphore.unlink()
            except Exception:
                pass
        finally:
            self.is_open = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
