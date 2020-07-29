def any_value(outputs):
    for key, value in outputs.items():
        if not key.startswith('precondition'):
            return value
