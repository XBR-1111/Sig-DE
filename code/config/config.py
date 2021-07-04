import json


def get_config():
    config_path = 'config/config.json'
    f = open(config_path, "r")
    json_obj = json.load(f)
    f.close()
    return json_obj
