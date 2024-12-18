
def seconds_to_human_readable(time):
    days = time // 86400  # (60 * 60 * 24)
    hours = time // 3600 % 24  # (60 * 60) % 24
    minutes = time // 60 % 60
    seconds = time % 60

    time_string = ""
    if days > 0:
        time_string += f"{days:.0f} day{'s' if days > 1 else ''}, "
    if hours > 0 or days > 0:
        time_string += f"{hours:02.0f}h:"
    time_string += f"{minutes:02.0f}m:{seconds:02.0f}s"

    return time_string
