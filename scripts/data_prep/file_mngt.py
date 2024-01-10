
# Saves lists to a folder path
def save_list_to_file(list_, folder_path_, filenm):
    with open(folder_path_+filenm, "w") as f:
        f.write("[\n")
        for element in list_:
            f.write(f'"{element}",\n')
        f.write("]")

