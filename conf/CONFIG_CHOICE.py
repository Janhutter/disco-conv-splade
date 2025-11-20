import os

CONFIG_NAME = None
CONFIG_PATH = "../conf"

##############################################################
# Provide (as env var), either:
# * 'SPLADE_CONFIG_NAME', this config in splade/conf will be used
# * or 'SPLADE_CONFIG_FULLPATH' (full path, from an exp, such as '/my/path/to/exp/config.yaml'

# if nothing is provided, 'config_default' is used
##############################################################

assert sum([v in os.environ.keys() for v in ["SPLADE_CONFIG_NAME", "SPLADE_CONFIG_FULLPATH"]]) <= 1

if "SPLADE_CONFIG_NAME" in os.environ.keys():
    raw_name = os.environ["SPLADE_CONFIG_NAME"]

    if raw_name.endswith(".yaml"):
        raw_name = raw_name[: -len(".yaml")]

    rel_dir, base_name = os.path.split(raw_name)
    if rel_dir:
        CONFIG_PATH = os.path.join(CONFIG_PATH, rel_dir)

    CONFIG_NAME = base_name
elif "SPLADE_CONFIG_FULLPATH" in os.environ.keys():
    CONFIG_FULLPATH = os.environ["SPLADE_CONFIG_FULLPATH"]
    CONFIG_PATH, CONFIG_NAME = os.path.split(CONFIG_FULLPATH)

    if CONFIG_NAME.endswith(".yaml"):
        CONFIG_NAME = CONFIG_NAME[: -len(".yaml")]
# elif 
else:
    CONFIG_NAME = "config_default"

if ".yaml" in CONFIG_NAME:
    CONFIG_NAME = CONFIG_NAME.split(".yaml")[0]
    CONFIG_NAME = CONFIG_NAME.split("/")[-1]
    CONFIG_PATH = CONFIG_PATH + "/" + "/".join(CONFIG_NAME.split("/")[:-1])

print(f"Using config: {CONFIG_NAME} from path: {CONFIG_PATH}")