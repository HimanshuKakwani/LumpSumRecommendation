import os
import sys

# Path to your project
project_home = '/home/yourusername/BONDS-SUGGESTIONS'
if project_home not in sys.path:
    sys.path.append(project_home)

# Use Mangum adapter so FastAPI runs as WSGI
from mangum import Mangum

# Import the FastAPI app from your script
from portfolio_ml_api import app

application = Mangum(app)
