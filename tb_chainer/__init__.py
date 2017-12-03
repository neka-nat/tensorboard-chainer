"""A module for visualization with tensorboard
"""

from .writer import FileWriter, SummaryWriter
from .record_writer import RecordWriter
from .name_scope import name_scope, within_name_scope, register_functions
from .graph import NodeName
