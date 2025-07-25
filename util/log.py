def log(str, path):
    print(str)
    with open(path, 'a') as f: 
        f.write(str + '\n') 


class Logger:
    def __init__(self, path):
        self.path = path

    def log(self, msg):
        print(msg)
        with open(self.path, 'a') as f:
            f.write(msg + '\n')

    def log_params(self, **kwargs):
        for key, value in kwargs.items():
            self.log(f"{key}: {value}")
        

def save_data(data, path):
    with open(path, 'a') as f: 
        f.write(str(data) + '\n') 