classFile = "./classes.txt"  # Class file
saveClassFile = "./classes_.txt"  # Class file

def load_class(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    return [x[:-1] for x in lines]


def write_class(file_path, classNames):
    file = open(file_path, 'w')
    file.write(str(classNames))
    file.write('\n'+str(len(classNames)))

classNames = load_class(classFile)  # load class name
print(classNames)
print(len(classNames))
write_class(saveClassFile, classNames)