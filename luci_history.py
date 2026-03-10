history = []

def log_command(command):
    history.append(command)
    if len(history) > 5:
        history.pop(0)

def get_history():
    return history