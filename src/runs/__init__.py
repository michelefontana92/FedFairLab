from .run_factory import RunFactory

from .FolkTablesBinary.folk_fairlab import *
from .CompasCentralized.compas_fairlab import *
from .MEPCentralized.mep_fairlab import *
from .CompasCentralized.compas_fedavg import * 
from .Employment.employment_fairlab import *
from .Insurance.insurance_fairlab import *
__all__=['RunFactory']