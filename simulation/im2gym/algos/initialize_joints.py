from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Process, Event
from simulation.im2gym.algos.pybullet_envs import createEnv
from pybullet_utils import bullet_client
from typing import Tuple
import numpy as np
import pybullet
import torch


RUN = 1
TERMINATE = -1

HAND = 8

def _set_joint_position(sim, robotId, numJoints, jointPosition):
    j = 0
    for i in range(numJoints):
        jointType = sim.getJointInfo(robotId, i)[2]
        if jointType == pybullet.JOINT_REVOLUTE:
            sim.resetJointState(robotId, i, jointPosition[j])
            j += 1

def _check_collision(sim, robotId):
    """
    Caution: This function returns True if the given joint position is IN COLLISION.
    """
    infeasible = False
    contactPoints = sim.getContactPoints(robotId)
    for i in range(len(contactPoints)):
        linkIndex = contactPoints[i][3]
        contactDistance = contactPoints[i][8]
        if linkIndex <= HAND and contactDistance < 0:
            infeasible = True
            break

    return infeasible

def collisionDetectionWorker(
    run, done, commandShared, envId: int, numWorkers: int, workerIndex: int, 
    jobAllocationInfoShared, jointPositionShared, initialObjectPoseShared
):
    import pybullet_data
    import pybullet

    command = np.ndarray(numWorkers, dtype=np.int8, buffer=commandShared.buf)
    jobAllocationInfo = np.ndarray((numWorkers, 3), dtype=np.int32, buffer=jobAllocationInfoShared.buf)
    batchSize = jobAllocationInfo[workerIndex, 0]
    jointPosition = np.ndarray((batchSize, 10), dtype=np.float64, buffer=jointPositionShared.buf)
    initialObjectPose = np.ndarray((batchSize, 7), dtype=np.float64, buffer=initialObjectPoseShared.buf)

    sim = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    robotId, numJoints, objectId = createEnv(envId, sim)

    done.set() # Signal that initialization is complete

    while True:
        run.wait() # Wait for run signal
        run.clear()
        if command[workerIndex] == TERMINATE: break # Terminate the process
        startIndex = jobAllocationInfo[workerIndex, 1]
        numSamples = jobAllocationInfo[workerIndex, 2]
        for i in range(numSamples):
            sim.resetBasePositionAndOrientation(
                objectId, initialObjectPose[(startIndex + i), :3], initialObjectPose[(startIndex + i), 3:]
            )
            _set_joint_position(sim, robotId, numJoints, jointPosition[(startIndex + i)])
            sim.performCollisionDetection()
            jointPosition[(startIndex + i), -1] = 1 - int(_check_collision(sim, robotId))
        done.set() # Signal that the job is done

    sim.disconnect()


class RandomInitialJointPosition:
    def __init__(self, envId: int, batchSize: int, numWorkers: int):
        self.envId = envId
        self.batchSize = batchSize
        self.numWorkers = numWorkers

        # Events to communicate with worker processes
        self.runs = list()
        for _ in range(self.numWorkers): self.runs.append(Event())
        self.dones = list()
        for _ in range(self.numWorkers): self.dones.append(Event())

        # Make shared memory used to communicate with worker processes
        self.smm = SharedMemoryManager()
        self.smm.start()
        self.commandShared, self.command = self._make_shared_memory(numWorkers, np.int8)
        self.jobAllocationInfoShared, self.jobAllocationInfo = self._make_shared_memory((numWorkers, 3), np.int32)
        self.initialObjectPoseShared, self.initialObjectPose = self._make_shared_memory(
            (self.batchSize, 7), np.float64
        )
        self.jointPositionShared, self.jointPosition = self._make_shared_memory(
            (self.batchSize, 10), np.float64
        )

        # Create pybullet workers
        self.workers = list()
        self.jobAllocationInfo[:, 0] = self.batchSize
        startIndices, jobAllocation = self._allocate_jobs(self.batchSize)
        self.jobAllocationInfo[:, 1] = startIndices
        self.jobAllocationInfo[:, 2] = jobAllocation
        for i in range(numWorkers):
            worker = Process(
                target=collisionDetectionWorker,
                args=(
                    self.runs[i], self.dones[i], self.commandShared, self.envId, self.numWorkers, i, 
                    self.jobAllocationInfoShared, self.jointPositionShared, self.initialObjectPoseShared
                )
            )
            worker.start()
            self.workers.append(worker)
        self._wait_workers() # Wait until the initialization is complete

    def _make_shared_memory(self, shape, dtype):
        arrayTemp = np.zeros(shape, dtype=dtype)
        arrayShared = self.smm.SharedMemory(size=arrayTemp.nbytes)
        array = np.ndarray(arrayTemp.shape, arrayTemp.dtype, arrayShared.buf)
        array[:] = arrayTemp[:]
        return arrayShared, array

    def _allocate_jobs(self, batchSize) -> Tuple[np.ndarray, np.ndarray]:
        jobAllocation = np.full(self.numWorkers, (batchSize / self.numWorkers), dtype=np.int32)
        remainingJobs = np.zeros_like(jobAllocation)
        remainingJobs[:(batchSize % self.numWorkers)] = 1
        jobAllocation += remainingJobs
        startIndices = np.roll(np.cumsum(jobAllocation), 1)
        startIndices[0] = 0
        return startIndices, jobAllocation

    def get_random_collision_free_joint_position(
        self, initialObjectPose: torch.Tensor, seed: int
    ) -> torch.Tensor:
        device = initialObjectPose.device
        self._set_job_allocation_info()
        initialObjectPose = self._to_numpy(initialObjectPose)
        jointPosition = self._sample_random_joint_position(seed)
        self._set_joint_position(jointPosition)
        self._set_initial_object_pose(initialObjectPose)
        self._run_workers()
        self._wait_workers()
        jointPositionAndFeasibility = self._get_joint_position_and_feasibility(device)
        return jointPositionAndFeasibility

    def _set_job_allocation_info(self):
        startIndices, jobAllocation = self._allocate_jobs(self.batchSize)
        self.jobAllocationInfo[:, 1] = startIndices
        self.jobAllocationInfo[:, 2] = jobAllocation

    def _to_numpy(self, initialObjectPose: torch.Tensor) -> np.ndarray:
        initialObjectPose = initialObjectPose.detach().cpu().numpy()
        return initialObjectPose

    def _sample_random_joint_position(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        jointMin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], dtype=np.float64)
        jointMax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], dtype=np.float64)
        jointPosition = rng.uniform(low=jointMin, high=jointMax, size=(self.batchSize, 9))
        return jointPosition

    def _set_joint_position(self, jointPosition: np.ndarray):
        self.jointPosition[:self.batchSize, :-1] = jointPosition[:]

    def _set_initial_object_pose(self, initialObjectPose: np.ndarray):
        self.initialObjectPose[:self.batchSize] = initialObjectPose[:]

    def _get_joint_position_and_feasibility(self, device) -> torch.Tensor:
        if device == 'cpu':
            return torch.Tensor(self.jointPosition[:self.batchSize], device=device)
        else:
            return torch.from_numpy(self.jointPosition[:self.batchSize]).to(device=device, dtype=torch.float)

    def _run_workers(self):
        self.command[:] = RUN
        for i in range(self.numWorkers): self.runs[i].set()
        
    def _wait_workers(self):
        for i in range(self.numWorkers): 
            self.dones[i].wait()
            self.dones[i].clear()
        
    def close(self):
        self.command[:] = TERMINATE
        for i in range(self.numWorkers): self.runs[i].set()
        for worker in self.workers:
            while worker.is_alive(): pass
        self.smm.shutdown()
