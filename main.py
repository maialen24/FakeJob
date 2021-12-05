
import subprocess
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   # exec('Preprocess.py')

    with open("./Output/outputPreprocess.txt", "w+") as output:
        subprocess.call(["python", "./Preprocess.py"], stdout=output);

    #exec('BASELINE.py')

    with open("./Output/outputBaseline.txt", "w+") as output:
        subprocess.call(["ipython", "./BASELINE.py"], stdout=output);

   # exec('MLP.py')

    with open("./Output/outputMLP.txt", "w+") as output:
        subprocess.call(["ipython", "./MLP.py"], stdout=output);