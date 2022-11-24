'''
Con este script vamos a poder leer las credentials de google
'''

import os
from base64 import b64decode
#
def main():
    key = os.environ.get("DRIVE_SECRET")
    key = str(key)
    with open("path.json", "w") as json_file:
        json_file.write(b64decode(key).decode())
    print(os.path.realpath("path.json"))

if __name__ == "__main__":
    main()
