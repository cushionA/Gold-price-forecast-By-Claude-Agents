import os
import sys

# Set environment variables
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'

# Change to project root
os.chdir(r'C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents')

# Import and run monitor v2
sys.path.insert(0, 'scripts')
from auto_resume_after_kaggle_v2 import KaggleMonitorV2

monitor = KaggleMonitorV2()
success = monitor.monitor()

sys.exit(0 if success else 1)
