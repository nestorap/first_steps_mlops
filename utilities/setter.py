'''
Con este script vamos a poder leer las credentials de google
'''

import os

def main():
    key = os.environ.get("SERVICE_ACCOUNT_KEY")
    with open("path.json", "w") as json_file:
        json_file.write(key)
    print(os.path.realpath(path.json))

if __name__ == "main":
    main()
