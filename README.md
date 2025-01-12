# Follow the huggingface reinforcement learning course

## Environment Setup
1. Setup system dependencies

```sh
sudo apt install -y swig cmake libsdl2-dev libfreetype6-dev python3-pip python3-venv python3-opengl ffmpeg xvfb
```

2. Setup venv with (remember to add `venv` folder into .gitignore)

```sh
python3 -m venv ./venv
source venv/bin/activate

echo "venv" >> .gitnore
echo "__pycache__" >> .gitignore
```

3. install python requirement
Note that `--no-build-isolation` is required to bypass the error when installing `gymnasium[box2d]` in a virtual environment.

```sh
pip install -r https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit1/requirements-unit1.txt

pip install -r requirements.txt --no-build-isolation
```

## For Unity Huggy project
Note that `ml-agents` only supports `python3.10`, my system `python3.12` (Ubuntu 24.04) can not install the dependencies properly due to the deprecation of `distutils`.

```sh
git clone --depth 1 https://github.com/Unity-Technologies/ml-agents

cd ml-agents
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
```

## Unit 2
```sh
pip install -r https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit2/requirements-unit2.txt
```
