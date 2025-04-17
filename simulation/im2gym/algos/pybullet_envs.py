from pybullet_utils import bullet_client
from abc import ABC, abstractmethod
from pathlib import Path
from enum import IntEnum
import pybullet_data
import numpy as np
import pybullet
import time


class Envs(IntEnum):
    Card = 0
    HiddenCard = 1
    FlipCard = 2
    Hole = 3
    Reorientation = 4
    Bookshelf = 5
    Bump = 6
    Hole_wide = 7


class EnvBase(ABC):
    def __init__(self):
        self.urdfDir = Path(__file__).parents[2].joinpath('assets', 'urdf', 'Panda')
        self.pointCloudDir = Path(__file__).parents[2].joinpath('assets', 'pointcloud')

    def _load_plane_and_robot(self, sim):
        sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Plane
        sim.loadURDF("plane.urdf")
        # Panda
        startPos = [0, 0, 0]
        startOrientation = sim.getQuaternionFromEuler([0, 0, 0])
        robotId = sim.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=1)
        numJoints = sim.getNumJoints(robotId)
        return robotId, numJoints

    @abstractmethod
    def loadEnv(self, sim):
        pass

    @abstractmethod
    def getObjectPointCloudPath(self) -> Path:
        pass


class CardEnv(EnvBase):
    def __init__(self):
        super().__init__()

    def loadEnv(self, sim):
        # Load plane and robot
        robotId, numJoints = self._load_plane_and_robot(sim)

        # Table
        tableShape = (0.2, 0.25, 0.2)
        tablePosition = (0.5, 0.0, 0.20)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Card
        cardShape = (0.025, 0.035, 0.0025)
        cardPosition = (0.5, 0.0, 0.4025)
        cardColor = (np.array([170, 170, 170, 255]) / 255.0).tolist()
        
        cardVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=cardShape,
            rgbaColor=cardColor
        )
        cardCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=cardShape
        )
        cardId = sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cardCollisionShapeId,
            baseVisualShapeIndex=cardVisualShapeId,
            basePosition=cardPosition
        )

        return robotId, numJoints, cardId
        
    def getObjectPointCloudPath(self) -> Path:
        return self.pointCloudDir.joinpath('card_pointcloud_32.npy')


class HiddenCardEnv(EnvBase):
    def __init__(self):
        super().__init__()

    def loadEnv(self, sim):
        # Load plane and robot
        robotId, numJoints = self._load_plane_and_robot(sim)

        # Table
        tableShape = (0.2, 0.25, 0.2)
        tablePosition = (0.5, 0.0, 0.2)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Card
        cardShape = (0.025, 0.035, 0.0025)
        cardPosition = (0.4, -0.15, 0.4025)
        cardColor = (np.array([255, 0, 0, 255]) / 255.0).tolist()
        cardVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=cardShape,
            rgbaColor=cardColor
        )
        cardCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=cardShape
        )
        cardId = sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cardCollisionShapeId,
            baseVisualShapeIndex=cardVisualShapeId,
            basePosition=cardPosition
        )

        table1Shape = (0.05, 0.125, 0.05)
        table1Position = (0.65, 0.0, 0.45)
        table1VisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=table1Shape
        )
        table1CollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=table1Shape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table1CollisionShapeId,
            baseVisualShapeIndex=table1VisualShapeId,
            basePosition=table1Position
        )

        table2Shape = (0.1, 0.125, 0.02)
        table2Position = (0.6, 0.0, 0.52)
        table2VisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=table2Shape
        )
        table2CollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=table2Shape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table2CollisionShapeId,
            baseVisualShapeIndex=table2VisualShapeId,
            basePosition=table2Position
        )
        
        return robotId, numJoints, cardId

    def getObjectPointCloudPath(self) -> Path:
        return self.pointCloudDir.joinpath('card_pointcloud_32.npy')


class FlipCardEnv(EnvBase):
    def __init__(self):
        super().__init__()

    def loadEnv(self, sim):
        # Load objects
        robotId, numJoints = self._load_plane_and_robot(sim)

        # Table
        tableShape = (0.2, 0.25, 0.2)
        tablePosition = (0.5, 0.0, 0.2)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Wall
        wallShape = (0.2, 0.025, 0.1)
        wallPosition = (0.5, 0.225, 0.5)
        wallVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=wallShape
        )
        wallCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=wallShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wallCollisionShapeId,
            baseVisualShapeIndex=wallVisualShapeId,
            basePosition=wallPosition
        )

        # Card
        cardShape = (0.025, 0.035, 0.0025)
        cardPosition = (0.4, -0.15, 0.4025)
        cardColor = (np.array([255, 0, 255, 255]) / 255.0).tolist()
        cardVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=cardShape,
            rgbaColor=cardColor
        )
        cardCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=cardShape
        )
        cardId = sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cardCollisionShapeId,
            baseVisualShapeIndex=cardVisualShapeId,
            basePosition=cardPosition
        )
      
        return robotId, numJoints, cardId

    def getObjectPointCloudPath(self) -> Path:
        return self.pointCloudDir.joinpath('card_pointcloud_32.npy')


class HoleEnv(EnvBase):
    def __init__(self):
        super().__init__()

    def loadEnv(self, sim):
        # Load objects
        robotId, numJoints = self._load_plane_and_robot(sim)

        # Hole
        holelength = 0.18

        # Table 1
        tableShape = (0.1-holelength/4, 0.25, 0.05)
        tablePosition = (0.7-(0.2-holelength/2)/2, 0.0, 0.35)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Table 2
        tableShape = (0.1-holelength/4, 0.25, 0.05)
        tablePosition = (0.3+(0.2-holelength/2)/2, 0.0, 0.35)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Table 3
        tableShape = (holelength/2, 0.125-holelength/4, 0.05)
        tablePosition = (0.5, 0.25-(0.25-holelength/2)/2, 0.35)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Table 4
        tableShape = (holelength/2, 0.125-holelength/4, 0.05)
        tablePosition = (0.5, -0.25+(0.25-holelength/2)/2, 0.35)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Table 5
        tableShape = (0.2, 0.25, 0.15)
        tablePosition = (0.5, 0.0, 0.15)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Card
        cardShape = (0.05, 0.1, 0.105)
        cardPosition = (0.62-0.05, 0.05, 0.19+0.105)
        cardColor = (np.array([255, 0, 0, 255]) / 255.0).tolist()
        cardVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=cardShape,
            rgbaColor=cardColor
        )
        cardCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=cardShape
        )
        cardId = sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cardCollisionShapeId,
            baseVisualShapeIndex=cardVisualShapeId,
            basePosition=cardPosition
        )
      
        return robotId, numJoints, cardId

    def getObjectPointCloudPath(self) -> Path:
        return self.pointCloudDir.joinpath('box_in_hole_pointcloud_128.npy')

class Hole_wideEnv(EnvBase):
    def __init__(self):
        super().__init__()

    def loadEnv(self, sim):
        # Load objects
        robotId, numJoints = self._load_plane_and_robot(sim)

        # Table 1
        tableShape = (0.2, 0.15, 0.16)
        tablePosition = (0.5, -0.05, 0.16)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Table 2
        tableShape = (0.2, 0.15, 0.2)
        tablePosition = (0.5, 0.25, 0.2)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Table 3
        # tableShape = (0.12, 0.0425, 0.105)
        # tablePosition = (0.5, 0.2075, 0.295)
        # tableVisualShapeId = sim.createVisualShape(
        #     shapeType=pybullet.GEOM_BOX,
        #     halfExtents=tableShape
        # )
        # tableCollisionShapeId = sim.createCollisionShape(
        #     shapeType=pybullet.GEOM_BOX, 
        #     halfExtents=tableShape
        # )
        # sim.createMultiBody(
        #     baseMass=0,
        #     baseCollisionShapeIndex=tableCollisionShapeId,
        #     baseVisualShapeIndex=tableVisualShapeId,
        #     basePosition=tablePosition
        # )

        # Table 4
        # tableShape = (0.12, 0.0425, 0.105)
        # tablePosition = (0.5, -0.2075, 0.295)
        # tableVisualShapeId = sim.createVisualShape(
        #     shapeType=pybullet.GEOM_BOX,
        #     halfExtents=tableShape
        # )
        # tableCollisionShapeId = sim.createCollisionShape(
        #     shapeType=pybullet.GEOM_BOX, 
        #     halfExtents=tableShape
        # )
        # sim.createMultiBody(
        #     baseMass=0,
        #     baseCollisionShapeIndex=tableCollisionShapeId,
        #     baseVisualShapeIndex=tableVisualShapeId,
        #     basePosition=tablePosition
        # )

        # Table 5
        # tableShape = (0.2, 0.25, 0.095)
        # tablePosition = (0.5, 0.0, 0.19)
        # tableVisualShapeId = sim.createVisualShape(
        #     shapeType=pybullet.GEOM_BOX,
        #     halfExtents=tableShape
        # )
        # tableCollisionShapeId = sim.createCollisionShape(
        #     shapeType=pybullet.GEOM_BOX, 
        #     halfExtents=tableShape
        # )
        # sim.createMultiBody(
        #     baseMass=0,
        #     baseCollisionShapeIndex=tableCollisionShapeId,
        #     baseVisualShapeIndex=tableVisualShapeId,
        #     basePosition=tablePosition
        # )

        # Card
        cardShape = (0.07, 0.01, 0.04)
        cardPosition = (0.5, 0, 0.32 + 0.04)
        cardColor = (np.array([255, 0, 0, 255]) / 255.0).tolist()
        cardVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=cardShape,
            rgbaColor=cardColor
        )
        cardCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=cardShape
        )
        cardId = sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cardCollisionShapeId,
            baseVisualShapeIndex=cardVisualShapeId,
            basePosition=cardPosition
        )
      
        return robotId, numJoints, cardId

    def getObjectPointCloudPath(self) -> Path:
        return self.pointCloudDir.joinpath('box_in_the_hole_wide_new_pointcloud_256.npy')


class ReorientationEnv(EnvBase):
    def __init__(self):
        super().__init__()

    def loadEnv(self, sim):
        # Load plane and robot
        robotId, numJoints = self._load_plane_and_robot(sim)

        # Bookshelf
        bumpposition = [0.0, -0.25, 0.0]
        startOrientation = sim.getQuaternionFromEuler([0, 0, 0])
        sim.loadURDF(str(self.urdfDir.joinpath('wall.urdf')), bumpposition, startOrientation, useFixedBase=1)

        # Box
        boxShape = (0.09, 0.09, 0.09)
        boxPosition = (0.5, 0.0, 0.445)
        boxColor = (np.array([170, 170, 170, 255]) / 255.0).tolist()
        boxVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=boxShape,
            rgbaColor=boxColor
        )
        boxCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=boxShape
        )
        boxId = sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=boxCollisionShapeId,
            baseVisualShapeIndex=boxVisualShapeId,
            basePosition=boxPosition
        )

        return robotId, numJoints, boxId

    def getObjectPointCloudPath(self) -> Path:
        return self.pointCloudDir.joinpath('box_9cm_pointcloud_32.npy')


class BookshelfEnv(EnvBase):
    def __init__(self):
        super().__init__()

    def loadEnv(self, sim):
        # Load objects
        robotId, numJoints = self._load_plane_and_robot(sim)

        # Bookshelf
        height = 0.4
        bookShape = (0.07, 0.02, 0.1)
        bookwidth = 0.01

        # 1. base
        tableShape = (0.1, 0.2, 0.2)
        tablePosition = (0.5, 0.0, 0.2)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # 2. left wall
        tableShape = (0.1, 0.01, height/2)
        tablePosition = (0.5, 0.2-0.01, 0.4+height/2)

        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # 3. right wall
        tableShape = (0.1, 0.01, height/2)
        tablePosition = (0.5, -0.2+0.01, 0.4+height/2)

        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # 4. back wall
        tableShape = (0.01, 0.2, height/2)
        tablePosition = (0.6-0.01, 0, 0.4+height/2)

        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # 5. shelf
        tableShape = (0.1, 0.2, 0.01)
        tablePosition = (0.5, 0.0, 0.4+height+0.01)

        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # 6. left book
        tableShape = bookShape
        tablePosition = (0.4+bookShape[0], 2*bookShape[1]+bookwidth, 0.4+bookShape[2])
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # 7. right book
        tableShape = bookShape
        tablePosition = (0.4+bookShape[0], -2*bookShape[1]-bookwidth, 0.4+bookShape[2])
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # 8. left left book
        tableShape = bookShape
        tablePosition = (0.4+bookShape[0], 4*bookShape[1]+2*bookwidth, 0.4+bookShape[2])
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # 9. right book
        tableShape = bookShape
        tablePosition = (0.4+bookShape[0], -4*bookShape[1]-2*bookwidth, 0.4+bookShape[2])
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Book
        bookPosition = (0.4+bookShape[0], 0.0, 0.4+bookShape[2])
        bookColor = (np.array([255, 0, 0, 255]) / 255.0).tolist()
        bookVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=bookShape,
            rgbaColor=bookColor
        )
        bookCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=bookShape
        )
        bookId = sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=bookCollisionShapeId,
            baseVisualShapeIndex=bookVisualShapeId,
            basePosition=bookPosition
        )

        return robotId, numJoints, bookId

    def getObjectPointCloudPath(self) -> Path:
        return self.pointCloudDir.joinpath('book_pointcloud_128.npy')


class BumpEnv(EnvBase):
    def __init__(self):
        super().__init__()

    def loadEnv(self, sim):
        # Load objects
        robotId, numJoints = self._load_plane_and_robot(sim)

        # Table
        tableShape = (0.2, 0.25, 0.2)
        tablePosition = (0.5, 0.0, 0.2)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Bump
        bumpHeight = 0.03
        tableShape = (0.2, 0.05, bumpHeight/2)
        tablePosition = (0.5, 0.0, 0.4+bumpHeight/2)
        tableVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=tableShape
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=tableShape
        )
        sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        # Box
        boxShape = (0.045, 0.045, 0.045)
        boxPosition = (0.5, 0.15, 0.445)
        boxColor = (np.array([170, 170, 170, 255]) / 255.0).tolist()
        boxVisualShapeId = sim.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=boxShape,
            rgbaColor=boxColor
        )
        boxCollisionShapeId = sim.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, 
            halfExtents=boxShape
        )
        boxId = sim.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=boxCollisionShapeId,
            baseVisualShapeIndex=boxVisualShapeId,
            basePosition=boxPosition
        )        
      
        return robotId, numJoints, boxId

    def getObjectPointCloudPath(self) -> Path:
        return self.pointCloudDir.joinpath('box_9cm_pointcloud_96.npy')


def _get_env(envId: int) -> EnvBase:
    envName = str(Envs(envId))[5:]
    env = eval(envName + 'Env')()
    return env


def createEnv(envId: int, sim):
    env = _get_env(envId)
    return env.loadEnv(sim)


def loadObjectPointCloud(envId: int) -> np.ndarray:
    env = _get_env(envId)
    return np.load(env.getObjectPointCloudPath())


if __name__ == "__main__":
    sim = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    createEnv(Envs.Hole_wide, sim)
    sim.performCollisionDetection()
    time.sleep(1000)
    sim.disconnect()
