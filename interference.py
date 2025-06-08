from roboflow import Roboflow

def get_model(model_id, api_key):
    rf = Roboflow(api_key=api_key)
    project_name, version = model_id.split('/')
    project = rf.workspace().project(project_name)
    model = project.version(int(version)).model
    return model