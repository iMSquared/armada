from simulation.im2gym.tasks.card import Card
from simulation.im2gym.tasks.bump import Bump
from simulation.im2gym.tasks.sysid import Sysid
from simulation.im2gym.tasks.domain_bimanual import DomainBimanual
from simulation.im2gym.tasks.throw_left import Throw_left


# Mappings from strings to environments
immgym_task_map = {
    "Card": Card, 
    "Bump": Bump,
    "Sysid": Sysid,
    "Throw": DomainBimanual,
    "Throw_left": Throw_left
}
