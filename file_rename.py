import os

categories = ["accordion", "airplanes", "bonsai", "buddha", "Faces", "pigeon", "rhino"]

for category in categories:
    directory = "images/" + category
    add_word = category + "_"
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        name, ext = os.path.splitext(filename)
        new_name = add_word + name + ext
        new_path = os.path.join(directory, new_name)
        os.rename(file_path, new_path)
