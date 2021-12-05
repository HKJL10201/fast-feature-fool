import os


def file_name(file_dir):
    with open("img_list1.txt", "w") as w:
        for root, dirs, files in os.walk(file_dir):
            files = sorted(files)
            for file in files:
                w.writelines(os.path.join(root, file) + "\n")


if __name__ == "__main__":
    file_name("/data/ltj/LTJ/common_data/ILSVRC2012_img_val")
