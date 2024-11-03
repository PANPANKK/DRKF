import sys


class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "a")  # 使用追加模式
        sys.stdout=self #重定向输出

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()

    def __del__(self):
        self.file.close()
        sys.stdout = self.terminal