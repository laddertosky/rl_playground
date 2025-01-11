# Follow the huggingface reinforcement learning course

## Environment Setup
1. setup system dependencies

```sh
sudo apt install -y swig cmake libsdl2-dev libfreetype6-dev python3-pip python3-venv python3-opengl ffmpeg xvfb
```

2. setup venv with (remember to add `venv` folder into .gitignore)

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


