import yaml 
import hiyapyco

def save_yaml(what, where):
    with open(where, 'w') as f:
        f.write(yaml.dump(what, default_flow_style = False))
        
def load_yaml(where):
    # with open(where, 'r') as f:
    #    config = hiyapyco.load(f, interpolate=True)
    config = hiyapyco.load(where, interpolate=True)

    return config