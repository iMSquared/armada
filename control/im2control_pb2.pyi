from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TorqueControlIn(_message.Message):
    __slots__ = ("arm", "desired_torques")
    ARM_FIELD_NUMBER: _ClassVar[int]
    DESIRED_TORQUES_FIELD_NUMBER: _ClassVar[int]
    arm: str
    desired_torques: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, arm: _Optional[str] = ..., desired_torques: _Optional[_Iterable[float]] = ...) -> None: ...

class PositionControlIn(_message.Message):
    __slots__ = ("arm", "desired_position", "desired_velocity", "desired_p_gains", "desired_d_gains")
    ARM_FIELD_NUMBER: _ClassVar[int]
    DESIRED_POSITION_FIELD_NUMBER: _ClassVar[int]
    DESIRED_VELOCITY_FIELD_NUMBER: _ClassVar[int]
    DESIRED_P_GAINS_FIELD_NUMBER: _ClassVar[int]
    DESIRED_D_GAINS_FIELD_NUMBER: _ClassVar[int]
    arm: str
    desired_position: _containers.RepeatedScalarFieldContainer[float]
    desired_velocity: _containers.RepeatedScalarFieldContainer[float]
    desired_p_gains: _containers.RepeatedScalarFieldContainer[float]
    desired_d_gains: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, arm: _Optional[str] = ..., desired_position: _Optional[_Iterable[float]] = ..., desired_velocity: _Optional[_Iterable[float]] = ..., desired_p_gains: _Optional[_Iterable[float]] = ..., desired_d_gains: _Optional[_Iterable[float]] = ...) -> None: ...

class JointPDControlIn(_message.Message):
    __slots__ = ("arm", "target_joint_position", "p_gains", "d_gains")
    ARM_FIELD_NUMBER: _ClassVar[int]
    TARGET_JOINT_POSITION_FIELD_NUMBER: _ClassVar[int]
    P_GAINS_FIELD_NUMBER: _ClassVar[int]
    D_GAINS_FIELD_NUMBER: _ClassVar[int]
    arm: str
    target_joint_position: _containers.RepeatedScalarFieldContainer[float]
    p_gains: _containers.RepeatedScalarFieldContainer[float]
    d_gains: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, arm: _Optional[str] = ..., target_joint_position: _Optional[_Iterable[float]] = ..., p_gains: _Optional[_Iterable[float]] = ..., d_gains: _Optional[_Iterable[float]] = ...) -> None: ...

class GripperControlIn(_message.Message):
    __slots__ = ("arm", "value")
    ARM_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    arm: str
    value: float
    def __init__(self, arm: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...

class JointState(_message.Message):
    __slots__ = ("current_position", "current_velocity", "current_torque", "current_gravity")
    CURRENT_POSITION_FIELD_NUMBER: _ClassVar[int]
    CURRENT_VELOCITY_FIELD_NUMBER: _ClassVar[int]
    CURRENT_TORQUE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_GRAVITY_FIELD_NUMBER: _ClassVar[int]
    current_position: _containers.RepeatedScalarFieldContainer[float]
    current_velocity: _containers.RepeatedScalarFieldContainer[float]
    current_torque: _containers.RepeatedScalarFieldContainer[float]
    current_gravity: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, current_position: _Optional[_Iterable[float]] = ..., current_velocity: _Optional[_Iterable[float]] = ..., current_torque: _Optional[_Iterable[float]] = ..., current_gravity: _Optional[_Iterable[float]] = ...) -> None: ...

class Pose(_message.Message):
    __slots__ = ("position", "orientation")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    position: _containers.RepeatedScalarFieldContainer[float]
    orientation: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, position: _Optional[_Iterable[float]] = ..., orientation: _Optional[_Iterable[float]] = ...) -> None: ...

class Jacobian(_message.Message):
    __slots__ = ("linear_jacobian", "angular_jacobian")
    LINEAR_JACOBIAN_FIELD_NUMBER: _ClassVar[int]
    ANGULAR_JACOBIAN_FIELD_NUMBER: _ClassVar[int]
    linear_jacobian: _containers.RepeatedScalarFieldContainer[float]
    angular_jacobian: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, linear_jacobian: _Optional[_Iterable[float]] = ..., angular_jacobian: _Optional[_Iterable[float]] = ...) -> None: ...

class NDArray(_message.Message):
    __slots__ = ("array",)
    ARRAY_FIELD_NUMBER: _ClassVar[int]
    array: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, array: _Optional[_Iterable[float]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Boolean(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: bool
    def __init__(self, value: bool = ...) -> None: ...

class String(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: str
    def __init__(self, value: _Optional[str] = ...) -> None: ...

class Float(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: float
    def __init__(self, value: _Optional[float] = ...) -> None: ...
