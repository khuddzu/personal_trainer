import os


def main():
    # Create the 'training_center' directory
    os.makedirs('training_center', exist_ok=True)

    # Create template directories and code
    create_directories()
    training_templates()
    models_fill()

def create_directories():
    os.makedirs('training_center/logs', exist_ok=True)
    os.makedirs('training_center/models', exist_ok=True)

def training_templates():
    

def models_fill():
    #Changes based on branch
