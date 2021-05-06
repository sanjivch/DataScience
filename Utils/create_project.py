import sys
import os
import shutil

# Copy Cookiecutter folder
source_folder_path = r'C:\Users\sanjiv\Documents\DataScience\Cookiecutter'

# Create a new folder
destination_path = r'C:\Users\sanjiv\Documents\DataScience\Projects'

# 
project_name = sys.argv[1]

# Create a copy of Cookiecutter with project name
shutil.copytree(source_folder_path, os.path.join(destination_path, project_name))
