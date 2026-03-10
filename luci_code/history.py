import json

commands = []

def add_command(command):
    commands.append(command)
    if len(commands) > 5:
        commands.pop(0)

def get_history():
    return commands

if __name__ == '__main__':
    print(get_history())