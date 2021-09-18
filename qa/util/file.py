import os
import sys
import json
import pickle


class FileUtil:
    def __init__(self, filepath: str, data: object = None):
        self.filepath = filepath
        self.data = data
        self.chunk_size = 256 * 1024 * 1024  # bytes

    def load(self):
        print("Loading file: {}".format(self.filepath))
        file_size = os.path.getsize(self.filepath)
        data = bytearray(0)
        with open(self.filepath, "rb") as f:
            for _ in range(0, file_size, self.chunk_size):
                data += f.read(self.chunk_size)
        return pickle.loads(data)

    def save(self, data: object = None):
        print("Saving file: {}".format(self.filepath))
        if data != None:
            self.data = data
        file = pickle.dumps(self.data)
        file_size = sys.getsizeof(file)  # bytes
        with open(self.filepath, "wb") as f:
            for index in range(0, file_size, self.chunk_size):
                f.write(file[index:(self.chunk_size + index)])
        return self

    def save_json(self, data: dict = None):
        print("Saving file: {}".format(self.filepath))
        file = open(self.filepath, "w")
        json.dump(data, file)
        file.close()
