def log(str, path):
    print(str)
    with open(path, 'a') as f: 
        f.write(str + '\n') 
        f.flush() 


class Logger:
    def __init__(self, path):
        self.path = path

    def log(self, msg):
        print(msg)
        with open(self.path, 'a') as f:
            f.write(msg + '\n')
            f.flush() 

    def log_params(self, **kwargs):
        for key, value in kwargs.items():
            self.log(f"{key}: {value}")
        

def save_data(data, path):
    with open(path, 'a') as f: 
        f.write(str(data) + '\n') 
        f.flush() 