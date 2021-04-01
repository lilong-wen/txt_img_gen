import yaml

with open('config.yaml') as conf_f:
    y = yaml.load(conf_f)
    print(y)
