import eel, os, subprocess
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

eel.init(os.path.join(BASE_DIR, 'ui'))

@eel.expose
def start_script(command):
    process = subprocess.Popen(
        "gnome-terminal -x {}".format(command), 
        stdout=subprocess.PIPE,
        stderr=None,
        shell=True
    )
    # os.system("gnome-terminal -- {}".format(command))

eel.start('index.html')