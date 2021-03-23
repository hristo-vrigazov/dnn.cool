

def unbind_task_labels(full_flow):
    for task_name, task in full_flow.get_all_children().items():
        task.labels = None
    return full_flow
