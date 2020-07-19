
def creat_one_folder_list(root, split='train'):  # like Jungle
    folder_root = os.path.join(root, split, 'image')
    filename = os.listdir(folder_root)     # find all images under the training set.
    f = open(os.path.join(root, (split + '.txt')), mode='w')

    for _i in filename:
        a, b = _i.split('img.png')
        f.write(a + '\n')

