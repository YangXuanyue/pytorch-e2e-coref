import configs

_log_file = open(f'{configs.logs_dir}/{configs.args.mode}.{configs.timestamp}.log', 'w')


def log(*args):
    print(*args)
    print(*args, file=_log_file, flush=True)


