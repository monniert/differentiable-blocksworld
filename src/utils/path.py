from pathlib import Path

# Project and source files
PROJECT_PATH = Path(__file__).parent.parent.parent
CONFIGS_PATH = PROJECT_PATH / 'configs'
DATASETS_PATH = PROJECT_PATH / 'datasets'
PRIMITIVES_PATH = PROJECT_PATH / 'primitives'
MODELS_PATH = PROJECT_PATH / 'models'
RUNS_PATH = PROJECT_PATH / 'runs'

# Other projects
NERFSTUDIO_PATH = PROJECT_PATH.parent / 'related_projects' / 'nerfstudio'
EMS_PATH = PROJECT_PATH.parent / 'related_projects' / 'EMS-superquadric_fitting' / 'Python' / 'src'
MBF_PATH = PROJECT_PATH.parent / 'related_projects' / 'MonteBoxFinder' / 'python'
