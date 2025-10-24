import os

def ensure_forward_slash(path):
    return path.replace("\\", "/")

def get_all_files(root, select_extensions=None, need_whole_path=True):
    filenames = os.listdir(root)

    select_file_paths = []
    for file_name in filenames:
        file_extension = os.path.splitext(file_name)[-1]

        if isinstance(select_extensions, list):
            if file_extension in select_extensions:
                if need_whole_path:
                    select_file_paths.append(ensure_forward_slash(os.path.join(root, file_name)))
                else:
                    select_file_paths.append(ensure_forward_slash(file_name))

        elif isinstance(select_extensions, str):
            if file_extension == select_extensions:
                if need_whole_path:
                    select_file_paths.append(ensure_forward_slash(os.path.join(root, file_name)))
                else:
                    select_file_paths.append(ensure_forward_slash(file_name))

    select_file_paths.sort()

    return select_file_paths


def mkdir_if_not_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
