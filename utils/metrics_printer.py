def print_metric_line(
        name, 
        metric,
        max_name_length,
        max_metric_length,
        first_line=False
):
    if not first_line:
        print("├─" + max_name_length*"─" + "─┼─" + max_metric_length*"─" + "─┤")
    str_metric = str(metric)
    print("│ " + (max_name_length-len(name))*" "  + f"{name} │ {str_metric}" + (max_metric_length-len(str_metric))*" " + " │")


def print_metrics(metrics):
    max_name_length = len("Name")
    max_metric_length = len("Value")
    for name, metric in metrics.items():
        max_name_length = max(max_name_length, len(name))
        max_metric_length = max(max_metric_length, len(str(metric)))

    # Top
    print("╭─" + max_name_length*"─" + "───" + max_metric_length*"─" + "─╮")
    # Title
    title_str = "Metrics"
    print("│ " + title_str + (max_name_length+max_metric_length+3-len(title_str))*" " + " │")
    # Header
    print("├─" + max_name_length*"─" + "─┬─" + max_metric_length*"─" + "─┤")
    print_metric_line(
        name="Name", 
        metric="Value",
        max_name_length=max_name_length,
        max_metric_length=max_metric_length,
        first_line=True
    )
    for name, metric in metrics.items():
        print_metric_line(
            name=name, 
            metric=metric,
            max_name_length=max_name_length,
            max_metric_length=max_metric_length
        )

    print("╰─" + max_name_length*"─" + "─┴─" + max_metric_length*"─" + "─╯")